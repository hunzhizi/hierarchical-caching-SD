import time

import torch
from torch.distributed import isend
from torch.package.analyze import find_first_use_of_broken_modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

from src.CacheManager import CacheManager
from src.Config import Config
from src.KVCacheModel import KVCacheModel
from src.util import seed_everything, norm_logits, sample, max_fn, greedy_sample
import torch.distributed as dist


class DecodingCpuCentric(ABC):
    def __init__(self,args):
        seed_everything(args.seed)
        self.args = args
        self.seed = args.seed
        self.seed_set = set()

        # 从 args 中获取分布式参数
        self.rank = args.rank  # 当前进程的全局排名
        self.world_size = args.world_size  # 总进程数
        self.local_rank = args.local_rank  # 本地设备上的排名（如单机多卡时为 GPU ID）

        self.is_drafter:bool = False
        self.is_target_model = False
        self.is_smallest_drafter = False
        # 初始化分布式进程组
        dist.init_process_group(
            backend='gloo',  # 如果是纯 CPU 用 'gloo'，GPU 建议 'nccl'
            init_method='tcp://127.0.0.1:12345',  # 或者从 args 传入
            rank=self.rank,
            world_size=self.world_size
        )
        self.color_print(f"初始化",self.rank)
        if self.local_rank != 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank - 1)
            # 注意 rank1 映射到 cuda:0
            self.device = torch.device(f"cuda:{self.local_rank - 1}")
            self.is_drafter = True
            if self.local_rank == 1:
                self.is_smallest_drafter = True
            if self.local_rank == self.world_size -1:
                self.is_target_model = True
                self.is_drafter = False

        elif self.local_rank == 0:
            # rank0 为 cpu 进行 负责集中通信
            self.device = torch.device("cpu")
        else :
            raise NotImplemented


        self.draft_forward_times:int = 0
        self.target_forward_times:int = 0
        self.num_acc_tokens:list = list()
        self.vocab_size = args.vocab_size
        self.eval_mode = args.eval_mode
        self.model_name = args.model_name
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.branch_prediction_num = args.branch_prediction_num
        self.drafters_num = len(args.draft_models_dir)
        self.target_model_rank = self.drafters_num

        self.verified_len = 0
        self.pending_verified_len = 0

    def load_model(self):
        # load models according to different evaluation methods.
        if self.args.eval_mode in ["default", "sd"]:
            self.color_print(f"Loading models:\n{self.args.draft_models_dir[0]}\n{self.args.target_model_dir}", 3)
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_models_dir[0], device_map="cuda:0",
                                                                    torch_dtype=torch.bfloat16,
                                                                    trust_remote_code=True).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model_dir,
                                                                     device_map="sequential",
                                                                     torch_dtype=torch.bfloat16,
                                                                     trust_remote_code=True).eval()
            # draft_forward_time初始化
            # self.draft_forward_time[self.args.draft_models_dir[0]] = 0
        elif self.eval_mode == "single_model" and self.model_name is not None:
            self.color_print(f"Loading models:{self.args.target_model_dir}\n", 3)
            # 单独测试模型的时候作为 target model 测试
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model_dir, device_map="cuda:0",
                                                              torch_dtype=torch.bfloat16,
                                                              trust_remote_code=True).eval()
        elif self.eval_mode in ["para_sd"]:
            print(f"进程 {self.local_rank} 的本地设备: {self.device}")
            if self.rank == 0:
                return
            # 所有模型仅用于推理，禁用梯度
            torch.set_grad_enabled(False)

            # 动态计算可用 GPU
            num_drafters = self.drafters_num
            num_total_gpus = torch.cuda.device_count()

            # 防御性检查
            assert num_total_gpus > num_drafters, "GPU 不足，无法隔离 Draft 和 Target 模型"

            if self.is_drafter:
                self.color_print(f"{self.device} Loading models:{self.args.draft_models_dir[self.local_rank - 1]}\n", self.rank)
                # Draft 模型：严格绑定到当前设备
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_models_dir[self.local_rank - 1],
                    device_map={"": self.device},  # 显式指定GPU索引
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
            if self.is_target_model:
                # Target 模型：自动分片到剩余 GPU
                target_gpus = list(range(num_drafters, num_total_gpus))
                self.color_print(f"Loading models:{self.args.target_model_dir}\n", self.rank)
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model_dir,
                    device_map="auto",
                    max_memory={gpu: "32GiB" for gpu in target_gpus},
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
                # 验证设备位置（调试用）
                # if self.local_rank < num_drafters:
                #     params = list(self.draft_model.parameters())
                #     assert all(p.device == self.accelerator.device for p in params), "Draft 模型设备错位！"
                # else:
                #     print(f"Target 模型设备映射: {self.target_model.hf_device_map}")

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_models_dir}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_models_dir[0], trust_remote_code=True)
        self.tokenizer.padding_side = "right"

        # for llama models
        self.tokenizer.pad_token_id = 2

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        # if self.accelerator.is_main_process:
        print(f"\033[9{color_number}m{content}\033[0m")

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad
    def autoregressive_sampling(self, prefix):
        if self.eval_mode != "single_model":
            raise RuntimeError("autoregressive_sampling only support single_model mode.")
        if self.model_name is None:
            raise RuntimeError("model_name is not specified.")
        model = self.target_model
        prefix = prefix.to(model.device)

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.max_tokens

        # x 为 在 decode 过程中生成的tokens shape: [batch_size, input_len]
        tokens_id = prefix
        past_key_values = None
        while tokens_id.shape[1] < max_tokens:
            if past_key_values is None:
                outputs = model(tokens_id)
            else:
                last_ids = tokens_id[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids.unsqueeze(0)
                outputs = model(last_ids, past_key_values=past_key_values, use_cache=True)
            self.target_forward_times += 1

            last_p = norm_logits(outputs.logits[:, -1, :], self.temperature, self.top_k, self.top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(max_fn(last_p))
            tokens_id = torch.cat([tokens_id, idx_next], dim=1)
        return tokens_id

    @torch.no_grad()
    def speculative_decoding(self, prefix):
        max_tokens = prefix.shape[1] + self.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        drafter_cache =KVCacheModel(self.draft_model, self.temperature, self.top_k, self.top_p, self.vocab_size)
        target_model_cache = KVCacheModel(self.target_model, self.temperature, self.top_k, self.top_p, self.vocab_size)

        while(prefix.shape[1] < max_tokens):
            prefix_len = prefix.shape[1]
            tokens_id =drafter_cache.generate(prefix.to(draft_device), self.branch_prediction_num)
            _ = target_model_cache.generate(tokens_id.to(target_device), 1)
            if self.rank==1:
                self.target_forward_times += 1
                self.draft_forward_times[self.args.draft_models_dir[0]] += self.branch_prediction_num

            n = prefix_len +self.branch_prediction_num - 1
            # 进行验证
            for i in range(self.branch_prediction_num):
                # 根据SD公式进行一个一个的验证
                r = torch.rand(1, device=draft_device)
                # 第j 列 token_id
                j = tokens_id[:, prefix_len + i]

                # 找到对应 token_id 的 probs进行验证
                if r > ( (target_model_cache.prob_history.to(draft_device)[:, prefix_len + i - 1, j])
                        /(drafter_cache.prob_history[:, prefix_len + i - 1, j])):
                    n = prefix_len + i -1
                    # i -1 就是接受的个数
                    break
            self.num_acc_tokens.append(n - prefix_len + 1)

            assert n >= prefix_len -1 ,f"成功验证后的最后位置索引n：{n}, prefix_len:{prefix_len}"
            prefix = tokens_id[:, :n + 1]
            # 回滚drafter
            drafter_cache.rollback(n + 1)

            # 回滚target model：判断 target model 接受 tokens 的个数
            # 1. 全部接受
            if n == prefix_len + self.branch_prediction_num - 1:
                # todo 这里 不用 max_fn 后再进行采样吗？ max_fn 使得所有的值 > 0 并且归一化
                next_token_id = sample(target_model_cache.prob_history[: , -1, :self.vocab_size]).to(draft_device)
                # next_token_id = sample(max_fn(target_model_cache.prob_history[: , -1, :self.vocab_size])).to(draft_device)
                # target_model_cache.rollback(n + 2) #todo 不用回滚吧
            # 2. 部分接受
            else:
                next_token_id = sample(max_fn(
                                target_model_cache.prob_history[:, n, :self.vocab_size].to(draft_device)
                                - drafter_cache.prob_history[:, n, :self.vocab_size] ))
                target_model_cache.rollback(n + 1)

            prefix = torch.cat([prefix, next_token_id], dim=1)
            print(f"generate is :{self.tokenizer.decode(prefix.squeeze())}")
        return prefix


    @torch.no_grad()
    def cpu_main(self):
        manager = CacheManager(self.world_size)
        manager.start()
        while True:
            time.sleep(100)
        #     print(f"[CPU] Current table size: {len(manager.table)}")


    @torch.no_grad()
    def gpu_main(self,prefix: torch.Tensor):
        if self.is_target_model:
            model = KVCacheModel(self.target_model, self.temperature, self.top_k, self.top_p, self.vocab_size).to(
                self.device)
        else:
            model = KVCacheModel(self.draft_model, self.temperature, self.top_k, self.top_p, self.vocab_size).to(
                self.device)
        prefix = prefix.to(self.device)

        max_tokens = Config.MAX_LEN
        if not self.is_target_model:
            max_tokens += 50

        # 创建通讯缓冲区
        recv_buffer = torch.full((1, Config.MAX_LEN), -1, dtype=torch.long, device='cpu')
        # 记录推理次数
        step: int = 0
        seq_len: int = 0
        # 统一向CPU发送数据
        dst = 0
        # while seq_len < max_tokens - 30:
        while seq_len < max_tokens:
            if seq_len == 0:
                # 直接进行推理
                prefix = model.generate(prefix, 1)
                # 更新 seq_len
                seq_len = prefix.shape[1]
                continue
            else:
                # 不是 prefill 阶段进行cache 的请求
                # decoding 过程
                # 先对 prefix 进行处理，然后进行通讯
                # currently only supports batch_size = 1
                self.color_print(f"\nrank{self.rank}向CacheManager发请求: \n {prefix} \n {self.tokenizer.decode(prefix[0].tolist())} \n", self.local_rank)
                send_buffer = prefix
                send_message = send_buffer.cpu()
                recv_message = recv_buffer.cpu()
                recv_message.fill_(-1)
                dist.send(tensor=send_message, dst=dst)
                dist.recv(tensor=recv_message, src=dst)
                send_buffer = send_message.to(self.device)
                recv_buffer = recv_message.to(self.device)
                input_ids = self.truncate_tensor(recv_buffer)
                self.color_print(f"收到的用于推理的tokens \n {input_ids} \n {self.tokenizer.decode(input_ids[0].tolist())} \n", self.local_rank)
                # 各个模型检查自己是否需要回滚
                index = self.find_first_diff_index(recv_buffer, send_buffer)
                if index < seq_len:
                    # 说明需要进行回滚
                    model.rollback(index)
                # 进行推理
                if self.is_smallest_drafter:
                    prefix = model.generate(input_ids, 1)
                    seq_len = prefix.shape[1]
                else:
                    # 其他模型
                    # 两种情况，
                    if index != seq_len or index == input_ids.shape[1]:
                        # 如果 recv_buffer 和 send_buffer 完全相同 则index == Config.MAX_LEN，此时没有携带candidates
                        # 1. 没有携带 candidates 直接进行推理
                        # self.color_print(f"收到的用于推理的tokens \n {input_ids} \n {self.tokenizer.decode(input_ids[0].tolist())} \n", self.local_rank)
                        prefix = model.generate(input_ids, 1)
                        seq_len = prefix.shape[1]
                    else:# 2.携带 candidates 推理后需要进行验证
                        # 目前只支持 batch_size = 1
                        pending_verification_tokens_id = model.generate(input_ids, 1)
                        cache_len:int = input_ids.shape[1]
                        # 进行验证
                        self.color_print(f'seq_len is {seq_len}', self.local_rank)
                        self.color_print(f"pending_verification_tokens_id shape[1] is {pending_verification_tokens_id.shape}", self.local_rank)
                        candidates = pending_verification_tokens_id[:, seq_len:-1]
                        predicted: torch.Tensor = model.prob_history[:, seq_len - 1:, :self.vocab_size].argmax(
                            dim=-1)
                        candidates = torch.hstack([candidates, predicted[:, -1:]])
                        self.color_print(f"predicted is {predicted}", self.local_rank)
                        self.color_print(f"candidates is {candidates}", self.local_rank)
                        verified = (predicted == candidates).all(dim=1)  # 检查所有Token是否匹配
                        if verified:
                            acc_token = cache_len - seq_len
                        else:
                            mismatch_pos = (predicted != candidates).nonzero(as_tuple=True)[1].min()
                            acc_token = mismatch_pos.item()
                        # 如果没有全部接受，模型回滚,
                        self.color_print(f"acc_token is {acc_token}")
                        if acc_token < cache_len - seq_len:
                            model.rollback(seq_len + acc_token )

                        # 更新 prefix
                        prefix = torch.cat([prefix,predicted[:, :acc_token + 1 ]],dim=-1)
                        seq_len += acc_token + 1

        if self.is_target_model:
            return prefix
        else:
            return None





    @staticmethod
    def truncate_tensor(tensor:torch.Tensor) -> torch.Tensor:
        # 检查输入张量是否为二维且第一维大小为 1
        if tensor.dim() == 2 and tensor.size(0) == 1:
            # 取出第一行
            row = tensor[0]
            # 找到第一个 -1 的索引，如果不存在则返回该行长度
            index = torch.argmax((row == -1).to(torch.int), dim=0)
            if row[index] != -1:
                return tensor
            # 截取第一行直到第一个 -1 的索引位置，并保持二维形状
            return row[:index].unsqueeze(0)
        else:
            raise ValueError("输入张量必须是形状为 (1, max_len) 的二维张量。")


    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            self.cpu_main()
        else:
            token_ids = self.gpu_main(prefix)
        # dist.barrier()
        return token_ids





    def find_first_diff_index(self,
                              a: torch.Tensor,
                              b: torch.Tensor) -> int:
        """
            查找两个张量在相同维度上第一个不同值的索引
            没有实现 维度为 (batch_size, seq_len 的版本)

            参数：
            a (torch.Tensor): 要比较的第一个张量，形状为 (1, seq_len)
            b (torch.Tensor): 要比较的第二个张量，形状为 (1, seq_len)

            返回：
            int: 第一个不同值的索引。如果完全相同，返回两者的最小长度

            异常：
            ValueError: 当输入张量维度不符合要求时抛出

            示例：
             a = torch.tensor([[1,2,3,4]])
             b = torch.tensor([[1,2,4,4]])
             find_first_diff_index(a, b)
            2
            """
        # 维度校验（允许单样本的2D张量）
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("输入张量应为二维，形状为 (1, seq_len)")

        # 确保比较的设备一致
        if a.device != b.device:
            b = b.to(a.device)

        # 获取最小比较长度
        min_len = min(a.size(1), b.size(1))

        # 提取有效切片（保持二维形状）
        a_slice = a[:, :min_len]
        b_slice = b[:, :min_len]

        # 生成差异掩码（保持维度）
        diff_mask = (a_slice != b_slice)

        # 查找第一个差异点
        if diff_mask.any():
            # 获取第一个True的二维坐标
            first_diff = (diff_mask).nonzero(as_tuple=True)[1].min()
            return first_diff.item()

        # 全部相同则返回最小长度
        return min_len




