import os
import re
import sys

from src.Config import Config

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import random
from src.util import seed_everything, parse_arguments
from src.DecodingCpuCentric import DecodingCpuCentric
import torch.distributed as dist

class EvalGSM8K(DecodingCpuCentric):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_name = "GSM8k"
        # organize cot(chain of thought) examples
        self.ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        self.INVALID_ANS = "[invalid]"
        self.ANSWER_TRIGGER = "The answer is"
        self.prompt = self.create_demo_text(ANSWER_TRIGGER=self.ANSWER_TRIGGER)

        # load relative resources
        self.load_tokenizer()
        self.load_data()
        self.load_model()
        dist.barrier()


    def create_demo_text(self, n_shot=8, cot_flag=True, ANSWER_TRIGGER="The answer is"):
        question, chain, answer = [], [], []
        question.append(
            "There are 15 trees in the grove. "
            "Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today?"
        )
        chain.append(
            "There are 15 trees originally. "
            "Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6."
        )
        answer.append("6")

        question.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot?"
        )
        chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        answer.append("5")

        question.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total?"
        )
        chain.append(
            "Originally, Leah had 32 chocolates. "
            "Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39."
        )
        answer.append("39")

        question.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
            "has 12 lollipops. How many lollipops did Jason give to Denny?"
        )
        chain.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some "
            "to Denny. So he gave Denny 20 - 12 = 8."
        )
        answer.append("8")

        question.append(
            "Shawn has five toys. For Christmas, he got two toys each from his "
            "mom and dad. How many toys does he have now?"
        )
        chain.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and "
            "dad, then that is 4 more toys. 5 + 4 = 9."
        )
        answer.append("9")

        question.append(
            "There were nine computers in the server room. Five more computers "
            "were installed each day, from monday to thursday. "
            "How many computers are now in the server room?"
        )
        chain.append(
            "There were originally 9 computers. For each of 4 days, 5 more "
            "computers were added. So 5 * 4 = 20 computers were added. "
            "9 + 20 is 29."
        )
        answer.append("29")

        question.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
            "wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?"
        )
        chain.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, "
            "he had 58 - 23 = 35. After losing 2 more, "
            "he had 35 - 2 = 33 golf balls."
        )
        answer.append("33")

        question.append(
            "Olivia has $23. She bought five bagels for $3 each. "
            "How much money does she have left?"
        )
        chain.append(
            "Olivia had 23 dollars. "
            "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
            "So she has 23 - 15 dollars left. 23 - 15 is 8."
        )
        answer.append("8")

        # randomize order of the examples ...
        index_list = list(range(len(question)))
        random.shuffle(index_list)

        # Concatenate demonstration examples ...
        demo_text = ""
        for i in index_list[:n_shot]:
            if cot_flag:
                demo_text += (
                    "Q: "
                    + question[i]
                    + "\nA: "
                    + chain[i]
                    + " "
                    + ANSWER_TRIGGER
                    + " "
                    + answer[i]
                    + ".\n\n"
                )
            else:
                demo_text += (
                    "Question: "
                    + question[i]
                    + "\nAnswer: "
                    + ANSWER_TRIGGER
                    + " "
                    + answer[i]
                    + ".\n\n"
                )
        return demo_text

    def extract_answer_from_output(self, completion):
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.INVALID_ANS

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading GSM8K data...", 3)
        data = []
        with open(os.path.join(self.args.data_path, "gsm8k.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["question"])
                encode_special_token_flag = not (
                            "Llama-3.2-1B-Instruct" in self.args.draft_models_dir and "Llama-3.1-8B-Instruct" in self.args.target_model)

                input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=encode_special_token_flag)
                datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                datum["ground_truth"] = self.extract_answer_from_output(datum["answer"])
                data.append(datum)
        self.data = data[:100]

        # random.shuffle(self.data)
        self.data = self.data

    def preprocess(self, input_text):
        text = self.prompt  + "Q: " + input_text + "\n" + "A:"
        return text

    def postprocess(self, input_text, output_text):
        bos_token_len = len(self.tokenizer.bos_token) if self.tokenizer.bos_token is not None else 0
        # generation = output_text[len(input_text)+len(self.tokenizer.bos_token)+1:] # tokenizer will add a '<s> ' at the beginning of the text.
        generation = output_text[len(input_text) + bos_token_len + 1:] # tokenizer will add a '<s> ' at the beginning of the text.
        generation = generation.lower()
        generation = generation.split(self.ANSWER_TRIGGER.lower())
        answer_flag = True if len(generation) > 1 else False
        if answer_flag:
            # Pick first answer with flag
            pred = generation[1]
        else:
            # Pick last number without flag
            pred = generation[-1]
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return self.INVALID_ANS

        if answer_flag:
            # choose the first element in list
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred[-1] == ".":
            pred = pred[:-1]

        return pred
             
    @torch.no_grad()
    def eval(self):
        if self.args.eval_mode == "sd" or self.args.eval_mode == "default":
            decoding = self.speculative_decoding
        elif self.args.eval_mode == "single_model":
            decoding = self.autoregressive_sampling
        elif self.args.eval_mode == "para_sd":
            decoding = self.parallel_speculative_decoding
        else:
            raise NotImplementedError

        out_path = os.path.join(self.args.exp_name, f"{self.args.eval_mode}_gsm8k.jsonl")
        out_f = open(out_path, "a")
        # wall_times = {"time":[], "num_tokens":[]}
        for _ in range(self.args.num_samples_per_task):
            # set random seed. Ensure each experiment runs with a unique random seed.
            while self.seed in self.seed_set:
                self.seed = random.randint(0, 1000000)
            seed_everything(self.seed)
            self.seed_set.add(self.seed)
            acc = 0
            for idx, datum in tqdm.tqdm(enumerate(self.data), total=len(self.data), disable=not self.is_target_model, ncols=50):
                input_ids = datum["input_ids"]
                torch.cuda.synchronize()
                start_time = time.time()
                generate_ids = decoding(input_ids)
                torch.cuda.synchronize()
                end_time = time.time()
                dist.barrier(self.gpu_group)
                if self.args.eval_mode == "para_sd":
                    if self.is_target_model:
                        dist.send(torch.tensor([self.rank], dtype=torch.int), dst=0, tag=Config.END_FLAG)
                    dist.barrier(self.gpu_group)
                if self.is_target_model:
                    # if idx != 0:
                        # skip the first prompt time consumption
                        # wall_times["time"].append(end_time-start_time)
                        # wall_times["num_tokens"].append(generate_ids.shape[1] - input_ids.shape[1])
                    answer = self.postprocess(datum["input_text"], self.tokenizer.decode(generate_ids[0, :]))
                    print(self.tokenizer.decode(generate_ids[0, :]))
                    # assert answer != self.INVALID_ANS, self.color_print(f"Invalid Answer!\n question:\n{datum['question']}\nanswer:\n{answer}", 1)
                    if answer == datum["ground_truth"]:
                        acc += 1
                    out_f.write(json.dumps({"question": datum["question"], "time": end_time-start_time, "new_tokens": generate_ids.shape[1] - input_ids.shape[1], "ground_truth": datum["ground_truth"], "answer": answer}, ensure_ascii=False) + "\n")
                out_f.flush()
            self.color_print(f"Accuracy: {acc / len(self.data):.4f} in the {_+1}-th iterations.", 2)
        
        out_f.close()
        
        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        # self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)
        
        # self.accelerator.wait_for_everyone()
        
        # if (self.accelerator.num_processes == 1 and self.accelerator.is_main_process) or (self.accelerator.num_processes == 2 and not self.accelerator.is_main_process):
        #     print(f"\033[92mtarget model forward times: {self.gather_forward_times()}\033[0m")
        
        # self.accelerator.wait_for_everyone()
        
        # if self.accelerator.is_main_process:
        #     speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
        #     speed_std = (torch.tensor(wall_times["num_tokens"]) / torch.tensor(wall_times["time"])).std().item()
        #     self.color_print(f"generate speed (tokens / second): {speed:.2f} with std {speed_std}", 2)


from transformers import AutoTokenizer, AutoModel
if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalGSM8K(args)
    alg.eval()
