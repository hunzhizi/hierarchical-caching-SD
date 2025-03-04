import time

from src.Decoding import Decoding
from src.util import parse_arguments
import torch
from time import perf_counter


class TestModel(Decoding):
    def __init__(self, args):
        super(TestModel, self).__init__(args)
        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()

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
        input_ids = self.tokenizer.encode("can you tell me a story about a man who is a detective and he is trying to solve a murder case. The story should be in the style of a noir novel. The detective is a private investigator named Jack who is hired by a wealthy businessman to solve a murder case. The businessman suspects that the victim, a wealthy businessman, was murdered by his own employees. ", add_special_tokens=encode_special_token_flag)
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
    args.eval_mode = "single_model"
    test_model = TestModel(args)
    test_model.eval()

def test_speculative_decoding():
    args = parse_arguments()
    args.eval_mode = "default"
    test_model = TestModel(args)
    test_model.eval()
    print(test_model.tokenizer.decode(token_ids=[4814, 498, 3291, 752, 264, 3364, 911, 264, 879, 879,
                                                 374, 264, 44159, 807, 807, 525, 4460]))
# todo 应该是回滚相关逻辑出现问题

if __name__ == '__main__':

    test_speculative_decoding()
    # test_single_model()
