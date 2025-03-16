import torch
import threading

class Notice:
    def __init__(self, max_len: int):
        self.input_ids: torch.Tensor = torch.full((1, max_len), -1, dtype=torch.long)
        self.is_update: bool = False
        self.update_cache: torch.Tensor = torch.full((1, max_len), -1, dtype=torch.long)
        self.recv_cache: torch.Tensor = torch.full((1, max_len), -1, dtype=torch.long)
        self.lock = threading.Lock()
        # todo 需要上锁访问吗？ 存在线程安全吗？