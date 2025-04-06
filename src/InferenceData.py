import json
from typing import Tuple
import os
from time import perf_counter


def save_dict_to_jsonl(data: dict, file_path: str):
    try:
        # 获取文件所在的目录
        dir_path = os.path.dirname(file_path)
        # 如果目录不存在，则创建目录
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "a", encoding="utf-8") as file:
            json_line = json.dumps(data, ensure_ascii=False)
            file.write(json_line + '\n')
    except Exception as e:
        print(f"保存文件的时候出现错误:{e}")



class InferenceData:
    """
    用于存储推理相关的信息
    注意在使用过程中要记录三个信息才能进正确使用该模块
    分别是：
    acc_len_list:list
    reject_len_list
    generate_timer
    """
    def __init__(self, rank: int = 0):
        self.rank: int = rank
        # 注意 这里的 acc_len 只包含 新生成的 token 原本大模型产生的token 不包含在内
        self.acc_len_list: list = list()
        self.reject_len_list: list = list()
        self.forward_time: int = 0
        self.exe_time: float = 0
        self.generate_timer = self.Timer()
        self.verification_timer = self.Timer()
        self.communication_timer = self.Timer()


    def reset_data(self):
        self.acc_len_list.clear()
        self.reject_len_list.clear()
        self.forward_time = 0
        self.exe_time = 0
        self.generate_timer.time_list.clear()
        self.verification_timer.time_list.clear()
        self.communication_timer.time_list.clear()

    def to_dict(self)-> dict:
        return self.__dict__

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
        generate_time = self.generate_timer.get_sum_time()
        verification_time = self.verification_timer.get_sum_time()
        communication_time = self.communication_timer.get_sum_time()
        self.exe_time = generate_time + verification_time + communication_time
        if verification_time == -1:
            self.exe_time += 1
        if communication_time == -1:
            self.exe_time += 1
        tokens_per_sec,mean_acc_len = self._get_tokens_per_second_and_mean_acc_len()
        data_view: dict = {
            "self.rank": self.rank,
            "tokens_per_sec":tokens_per_sec,
            "mean_acc_len": mean_acc_len,
            "exe_time": self.exe_time,
            "generate_time": generate_time,
            "verification_time": verification_time,
            "communication_time": communication_time,
            "forward_time": self.forward_time,
            "acc_len_list": self.acc_len_list,
            "reject_len_list": self.reject_len_list,
        }
        if is_store:
            save_dict_to_jsonl(data_view,file_path=file_path)
        if is_reset:
            self.reset_data()
        return data_view

    class Timer:
        def __init__(self):
            self.time_list: list = list()


        def __enter__(self):
            self.start_time = perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = perf_counter()
            self.execution_time = self.end_time - self.start_time
            self.time_list.append(self.execution_time)

        def get_sum_time(self) -> float:
            if len(self.time_list) == 0:
                # -1 表示没有记录该项
                return -1
            return sum(self.time_list)
