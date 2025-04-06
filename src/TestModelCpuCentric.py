import time

from src.Config import Config
from src.DecodingCpuCentric import DecodingCpuCentric
from src.util import parse_arguments
import torch
from time import perf_counter, sleep
import torch.distributed as dist
import sys
import os

class TestModelCpuCentric(DecodingCpuCentric):
    def __init__(self, args):
        super(TestModelCpuCentric, self).__init__(args)
        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()
        dist.barrier()

    def load_data(self):
        pass

    def eval(self):
        if self.args.eval_mode == "sd" or self.args.eval_mode == "default":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "single_model":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        else:
            raise NotImplementedError("Not implemented yet.")
        encode_special_token_flag = not ("Llama-3.2-1B-Instruct" in self.args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.args.target_model)
        for i in range(10):
            input_ids = self.tokenizer.encode("Imagine a world where gravity reverses every 24 hours. Describe its societal, scientific, and architectural consequences.",
                                              add_special_tokens=encode_special_token_flag)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            start = perf_counter()
            generate_ids = decoding(input_ids)
            end = perf_counter()

            if self.is_target_model:
                print(f"第{i + 1}次推理")
                print(f"精确耗时：{(end - start) * 1000:.3f} 毫秒")
                generate_ids = self.tokenizer.decode(generate_ids.squeeze())
                print(f"generate_ids: {generate_ids}")
            # 在这里同步，通知cache manager 进行cache manage的 reset
            dist.barrier(self.gpu_group)
            if self.args.eval_mode == "para_sd":
                if self.is_target_model:
                    dist.send(torch.tensor([self.rank], dtype=torch.int), dst=0, tag=Config.END_FLAG)
                dist.barrier(self.gpu_group)


    def postprocess(self, input_text, output_text):
        pass

    def preprocess(self, input_text):
        pass


def test_speculative_decoding():
    args = parse_arguments()
    # args.eval_mode = "default"
    test_model = TestModelCpuCentric(args)
    test_model.eval()


if __name__ == '__main__':
    test_speculative_decoding()
    # test_single_model()

