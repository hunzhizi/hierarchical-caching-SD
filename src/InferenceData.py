import json
from typing import Tuple
import os


def save_dict_to_jsonl(data: dict, file_path: str):
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            json_line = json.dumps(data,ensure_ascii=False)
            file.write(json_line + '\n')
    except Exception as e:
        print(f"保存文件的时候出现错误:{e}")


class InferenceData:
    """
    用于存储推理相关的信息
    todo 后续可能添加 内部类增加函数执行时间测试模块
    """
    def __init__(self):
        # 注意 这里的 acc_len 只包含 新生成的 token 原本大模型产生的token 不包含在内
        self.acc_len_list: list = list()
        self.reject_len_list: list = list()
        self.forward_time: int = 0
        self.exe_time: float = 0

    def reset_data(self):
        self.acc_len_list.clear()
        self.reject_len_list.clear()
        self.forward_time = 0
        self.exe_time = 0

    def _get_tokens_per_second_and_mean_acc_len(self) -> Tuple[float,float]:
        if self.exe_time == 0:
            raise RuntimeError(f"you need to set exe_time,current exe_time is{self.exe_time}")
        all_acc_tokens = [i + 1 for i in self.acc_len_list]
        generate_tokens_num = sum(all_acc_tokens)
        self.forward_time = len(all_acc_tokens)
        mean_acc_len = generate_tokens_num/len(all_acc_tokens)
        return generate_tokens_num/self.exe_time, mean_acc_len

    def add_acc(self,acc_num: int):
        self.acc_len_list.append(acc_num)

    def add_reject(self, acc_num: int):
        self.reject_len_list.append(acc_num)


    def get_inference_data(self,
                           is_store: bool = False,
                           is_reset: bool = True,
                           file_path: str = None):
        tokens_per_sec,mean_acc_len = self._get_tokens_per_second_and_mean_acc_len()
        data_view: dict = {
            "acc_len_list": self.acc_len_list,
            "reject_len_list": self.reject_len_list,
            "forward_time": self.forward_time,
            "exe_time": self.exe_time,
            "tokens_per_sec":tokens_per_sec,
            "mean_acc_len": mean_acc_len,
        }
        if is_store:
            save_dict_to_jsonl(data_view,file_path=file_path)
        if is_reset:
            self.reset_data()
        return data_view
