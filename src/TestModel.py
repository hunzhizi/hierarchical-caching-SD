import time

from src.Decoding import Decoding
from src.util import parse_arguments
import torch
from time import perf_counter
import torch.distributed as dist
import sys
import os

class TestModel(Decoding):
    def __init__(self, args):
        super(TestModel, self).__init__(args)
        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()
        self.accelerator.wait_for_everyone()

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
        encode_special_token_flag = not ("Llama-3.1" in self.args.draft_models_dir and "Llama-3.1" in self.args.target_model)
        input_ids = self.tokenizer.encode("Write Act II of a Shakespearean - style tragedy titled The Shadow Throne. Scene 1: A royal council meeting where conflicting prophecies emerge. Scene 2: A secret rendezvous between the prince and his forbidden lover. Scene 3: The assassination attempt that changes everything."
                                          , add_special_tokens=encode_special_token_flag)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        start = perf_counter()
        generate_ids = decoding(input_ids)
        end = perf_counter()
        print(f"精确耗时：{(end - start) * 1000:.3f} 毫秒")
        generate_ids = self.tokenizer.decode(generate_ids.squeeze())
        print(f"generate_ids: {generate_ids}")

    def postprocess(self, input_text, output_text):
        pass

    def preprocess(self, input_text):
        pass

def test_single_model():
    args = parse_arguments()
    args.max_tokens = 1000
    args.eval_mode = "single_model"
    args.target_model = "Qwen2.5-7B-Instruct"
    test_model = TestModel(args)
    test_model.eval()

def test_speculative_decoding():
    args = parse_arguments()
    # args.eval_mode = "default"
    test_model = TestModel(args)
    test_model.eval()


if __name__ == '__main__':

    # test_speculative_decoding()
    test_single_model()
