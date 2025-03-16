import torch
from torch.distributed import isend
from torch.package.analyze import find_first_use_of_broken_modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from accelerate import Accelerator

from src.Config import Config
from src.KVCacheModel import KVCacheModel
from src.util import seed_everything, norm_logits, sample, max_fn, greedy_sample
import torch.distributed as dist


class Decoding(ABC):
    def __init__(self,args):
        seed_everything(args.seed)
        self.args = args
        self.seed = args.seed
        self.seed_set = set()
        self.accelerator = Accelerator()

        # todo 进程数量 根据 args.eval_mode 进行断言

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
        self.is_target_model = False
        self.is_smallest_drafter = False
        self.local_rank = self.accelerator.local_process_index
        self.device = self.accelerator.device  # 直接绑定设备
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
            print(f"进程 {self.accelerator.local_process_index} 的本地设备: {self.accelerator.device}")

            # 确保 accelerator 已初始化
            if not hasattr(self, 'accelerator'):
                self.accelerator = Accelerator()
            # 所有模型仅用于推理，禁用梯度
            torch.set_grad_enabled(False)

            # 动态计算可用 GPU
            num_drafters = len(self.args.draft_models_dir)
            num_total_gpus = torch.cuda.device_count()

            # 防御性检查
            assert num_total_gpus > num_drafters, "GPU 不足，无法隔离 Draft 和 Target 模型"

            if self.accelerator.local_process_index < num_drafters:
                self.color_print(f"{self.accelerator.device} Loading models:{self.args.draft_models_dir[self.accelerator.local_process_index]}\n", 3)
                # Draft 模型：严格绑定到当前设备
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_models_dir[self.accelerator.local_process_index],
                    # device_map={"": self.accelerator.device},  # 关键修改
                    device_map={"": self.accelerator.device},  # 显式指定GPU索引
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
                if self.local_rank == 0:
                    self.is_smallest_drafter = True

            else:
                # Target 模型：自动分片到剩余 GPU
                target_gpus = list(range(num_drafters, num_total_gpus))
                self.color_print(f"Loading models:{self.args.target_model_dir}\n", 2)
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model_dir,
                    device_map="auto",
                    max_memory={gpu: "32GiB" for gpu in target_gpus},
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval()
                self.is_target_model = True
                # 验证设备位置（调试用）
                if self.accelerator.local_process_index < num_drafters:
                    params = list(self.draft_model.parameters())
                    assert all(p.device == self.accelerator.device for p in params), "Draft 模型设备错位！"
                else:
                    print(f"Target 模型设备映射: {self.target_model.hf_device_map}")

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
            if self.accelerator.is_main_process:
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
    def parallel_speculative_decoding(self, prefix: torch.Tensor) -> torch.Tensor:
        # parallel speculative decoding todo 修改成多机环境
        if self.is_target_model:
            model = KVCacheModel(self.target_model, self.temperature, self.top_k, self.top_p, self.vocab_size).to(self.device)
        else:
            model = KVCacheModel(self.draft_model, self.temperature, self.top_k, self.top_p, self.vocab_size).to(self.device)
        prefix.to(self.device)
        max_tokens = prefix.shape[1] + self.max_tokens
        # state[0] 的值表示 sender 的 accelerate.rank
        # state[0]: needs to be updated, transfered by other bigger drafters or target model
            # the value in state[0] is the sender rank
        # state[1]: is branch prediction.
        # 每个进程都在当前进程创建一个状态变量用于进行流控制
        # state = torch.tensor([-1,-1],device=self.accelerator.device)
        tokens_cache_req0: torch.Tensor = torch.zeros(max_tokens, device=self.accelerator.device, dtype=torch.int64).unsqueeze(0)
        tokens_cache_req1: torch.Tensor = torch.zeros(max_tokens, device=self.accelerator.device, dtype=torch.int64).unsqueeze(0)
        # req_list: [0]: target model notice [1]: next model notice
        req_list: list = list()
        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            input_ids = prefix.to(self.accelerator.device)
            self.color_print(f"开始执行{self.local_rank}",self.local_rank)
            if self.is_smallest_drafter:
                # 有两个接受的req 一个是 target model ，一个是比他大一个rank 的model
                if len(req_list) == 0:  # 第一次推理
                    # 创建两个req 分别接受 target model 和 bigger one
                    # todo 要判断 if self.drafter_num  > 2, if not create one req should be enough
                    self.color_print(f"进入{self.local_rank}", self.local_rank)
                    print(f"dist.Backend.NCCL:{dist.get_backend()}")
                    req_target_model = dist.irecv(tokens_cache_req0, self.drafters_num, tag=Config.RECEIVE_TOKENS)
                    print(tokens_cache_req0)
                    req_next_model = dist.irecv(tokens_cache_req1, self.local_rank + 1, tag=Config.RECEIVE_TOKENS)

                    req_list.append(req_target_model)
                    req_list.append(req_next_model)

                    prefix = model.generate(input_ids, 1)
                    self.color_print(f"{self.device}: generate{prefix}",self.local_rank)
                    self.draft_forward_times += 1
                else: # 不是第一次推理
                    if req_list[0].is_completed():
                        # target model notice: 2 cases
                        # 1. notice update
                        # 2. notice update and branch prediction
                        # currently I didn't use tree cache ,so the notice must be updating and branch prediction
                        # tree cache would be different
                        # update
                        prefix = tokens_cache_req0[:, :].clone()
                        prefix_len = prefix.shape[1]
                        # 全部接受 -》 不会 进行更新
                        # 部分接受 -》 某部分被拒绝，会进行更新，并且会 sample 出一个新的token ，所以要进行深拷贝，并且回滚到最长的验证长度
                        model.rollback(self.verified_len)
                        self.verified_len = prefix_len
                        self.pending_verified_len = prefix_len
                        # branch_prediction
                        prefix = model.generate(input_ids, Config.PREDICTION_NUM)
                        self.color_print(f"{self.device}: generate {prefix}", 1)
                        self.draft_forward_times += Config.PREDICTION_NUM
                        dist.isend(prefix, self.target_model_rank, tag=Config.SEND_TOKENS)
                        dist.isend(prefix, self.local_rank + 1, tag=Config.SEND_TOKENS)
                        req_list[0] = dist.irecv(tokens_cache_req0, self.drafters_num, tag=Config.RECEIVE_TOKENS)
                        continue
                    if req_list[1].is_completed():
                        # next model notice
                        # 检查下一层的 cache 是否命中
                        cache_len = self.find_first_diff_index(tokens_cache_req1, input_ids) + 1
                        req_list[1] = dist.irecv(tokens_cache_req1, self.local_rank + 1, tag=Config.RECEIVE_TOKENS)

                        if cache_len == tokens_cache_req1[0].shape[1]:
                            # cache 全部命中
                            self.pending_verified_len = cache_len
                        else:
                            prefix = tokens_cache_req1[:, :].clone()
                            prefix_len = prefix.shape[1]
                            model.rollback(self.pending_verified_len)
                            self.pending_verified_len = prefix_len
                            continue

                    # 进行正常推理
                    prefix = model.generate(input_ids, 1)
                    self.color_print(f"{self.device}: generate{prefix}",1)
                    self.draft_forward_times += 1

            elif self.is_target_model:

                if self.verified_len == 0:
                    # 第一次推理
                    # 直接生成 tokens, 更新 prefix
                    prefix = model.generate(input_ids, 1)
                    self.color_print(f"{self.device}: generate{prefix}",2)
                    self.target_forward_times += 1
                    self.verified_len = prefix.shape[1]

                else:
                    # 不是第一次推理
                    # 如果不是第一次推理，
                    # 通知上层model发送cache给 target model，查询cache，
                    # 携带candidates推理，进行 candidates 验证,目前只使用 greedy decoding
                    dist.isend(prefix, self.local_rank - 1, tag=Config.SEND_TOKENS)
                    dist.recv(tokens_cache_req0, self.local_rank - 1, tag=Config.RECEIVE_TOKENS)
                    cache_len = self.find_first_diff_index(tokens_cache_req0, input_ids) + 1
                    # 在上层去更新cache
                    if cache_len <= prefix_len:
                        # 说明 cache 未命中，需要进行 分支预测。
                        # 通知除了上层模型外的模型进行更新，
                        # 等待分支预测模型的candidates 这里目前使用最小的模型做分支预测
                        for i in range(self.drafters_num - 1):
                            # 这里选用最小的模型进行分支预测
                            dist.isend(prefix, i, tag=Config.SEND_TOKENS)
                        # 等待 分支预测模型返回cache
                        dist.recv(tokens_cache_req0, 0,tag=Config.RECEIVE_TOKENS)


                    # 携带 candidates 进行 generate

                    prefix = tokens_cache_req0[:,:].clone()
                    # 目前只支持 batch_size = 1
                    pending_verification_tokens_id=model.generate(prefix,1)[0]
                    self.color_print(f"{self.device}: generate{prefix}",2)
                    # 进行greedy decoding验证
                    # 总共验证的 tokens 的个数为： predicted_len - prefix_len
                    # acc_token = 0
                    # # 注意记录 prefix
                    # t = greedy_sample(model.prob_history[:, prefix_len , :self.vocab_size])
                    # prefix = torch.cat([prefix, t], dim=1)
                    # for i in range(1, cache_len - prefix_len):
                    #     t = greedy_sample(model.prob_history[:, prefix_len + i, :self.vocab_size])
                    #     prefix = torch.cat([prefix, t], dim=1)
                    #     if t != pending_verification_tokens_id[:,prefix_len + i]:
                    #         break
                    #     else:
                    #         acc_token+=1
                    # 优化为批量比较
                    candidates = pending_verification_tokens_id[:, prefix_len:cache_len]
                    predicted: torch.Tensor = model.prob_history[:, prefix_len:cache_len, :self.vocab_size].argmax(dim=-1)
                    verified = (predicted == candidates).all(dim=1)  # 检查所有Token是否匹配
                    if verified:
                        acc_token = cache_len - prefix_len
                    else:
                        mismatch_pos = (predicted != candidates).nonzero(as_tuple=True)[1].min()
                        acc_token = mismatch_pos.item()
                    # 如果没有全部接受，模型回滚,
                    # todo 可以提前通知其他所有模型进行回滚，和提前进行分支预测提高效率
                    if acc_token < cache_len - prefix_len:
                        model.rollback(prefix_len + acc_token + 1)
            else:   # 这种情况属于中间的 drafters
                # 最大的模型和下一个模型可以通知这些 drafters 进行更新
                if self.verified_len == 0:
                    # 第一次推理
                    self.verified_len = prefix_len
                    if self.local_rank + 1 == self.drafters_num:
                        req_target_model = dist.irecv(tokens_cache_req0, self.drafters_num, tag=Config.RECEIVE_TOKENS)
                        req_list.append(req_target_model)

                    req_next_model = dist.irecv(tokens_cache_req1, self.local_rank + 1, tag=Config.RECEIVE_TOKENS)
                    req_list.append(req_next_model)

                    prefix = model.generate(input_ids, 1)
                    self.color_print(f"{self.device}: generate: {prefix}",3)
                    self.target_forward_times += 1

                else:
                    # 不是第一次推理
                    # 通知上层 model 发送 cache 给 target model 查询 cache
                    # 如果能够携带 candidiates 进行推理就携带，不能就直接进行推理

                    # 首先查看 target model 有没有通知更新
                    if self.local_rank != self.drafters_num-1 and req_list[0].is_completed():
                        prefix = tokens_cache_req0[:, :].clone()
                        prefix_len = prefix.shape[1]
                        # 全部接受 -》 不会 进行更新
                        # 部分接受 -》 某部分被拒绝，会进行更新，并且会 sample 出一个新的token ，所以要进行深拷贝，并且回滚到最长的验证长度
                        model.rollback(self.verified_len)
                        self.verified_len = prefix_len
                        self.pending_verified_len = prefix_len
                        # 直接进行 generate
                        model.generate(prefix, 1)
                        self.color_print(f"{self.device}: generate{prefix}", 3)
                        self.draft_forward_times += 1
                        req_list[0] = dist.irecv(tokens_cache_req0, self.target_model_rank, tag=Config.RECEIVE_TOKENS)
                        continue
                    if req_list[1].is_completed():
                        # 进行验证
                        # next model notice
                        # 检查下一层的 cache 是否命中
                        cache_len = self.find_first_diff_index(tokens_cache_req1, input_ids) + 1
                        req_list[1] = dist.irecv(tokens_cache_req1, self.local_rank + 1, tag=Config.RECEIVE_TOKENS)

                        if cache_len == tokens_cache_req1[0].shape[1]:
                            # cache 全部命中
                            self.pending_verified_len = cache_len
                        else:
                            # 未全名中需要进行回滚
                            prefix = tokens_cache_req1[:, :].clone()
                            prefix_len = prefix.shape[1]
                            model.rollback(self.pending_verified_len)
                            self.pending_verified_len = prefix_len
                            continue

                    # 向上层 model 索要 cache
                    dist.isend(prefix, self.local_rank -1 , tag=Config.SEND_TOKENS)
                    dist.recv(tokens_cache_req1, self.local_rank - 1, tag=Config.RECEIVE_TOKENS)
                    cache_len = self.find_first_diff_index(tokens_cache_req1, input_ids) + 1
                    if cache_len <= prefix_len:
                        # 不携带 cache 推理
                        model.generate(prefix, 1)
                        self.color_print(f"{self.device}: generate{prefix}", 3)
                        self.draft_forward_times += 1

                    else: # cache 命中 携带 cache 进行推理
                        prefix = tokens_cache_req1[:, :cache_len].clone()
                        pending_verification_tokens_id = model.generate(prefix, 1)
                        self.color_print(f"{self.device}: generate{prefix}", 3)
                        self.draft_forward_times += 1
                        # 进行 greedy decoding 验证
                        # 优化为批量比较
                        candidates = pending_verification_tokens_id[:, prefix_len:cache_len]
                        predicted = model.prob_history[:, prefix_len:cache_len, :self.vocab_size].argmax(dim=-1)
                        verified = (predicted == candidates).all(dim=1)  # 检查所有Token是否匹配
                        if verified:
                            acc_token = cache_len - prefix_len
                        else:
                            mismatch_pos = (predicted != candidates).nonzero(as_tuple=True)[1].min()
                            acc_token = mismatch_pos.item()

                        if acc_token < cache_len - prefix_len:
                            model.rollback(prefix_len + acc_token + 1)















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




