import torch
import threading
import torch.distributed as dist
from conda.notices import notices

from src.Config import Config
from src.Notice import Notice
from typing import List


class CacheManager:
    def __init__(self, world_size,max_len = Config.MAX_LEN):
        self.world_size = world_size
        self.lock = threading.Lock()
        self.recv_threads = []
        # in fact , we only need ${world_size} notice, but for easier indexing, we use world_size+1
        self.max_len: int = max_len
        self.notices: List[Notice] = [Notice(max_len) for i in range(world_size + 1)]
        self.global_condition = threading.Condition()  # 全局条件变量


    def start(self):
        """启动接收线程处理所有GPU进程"""

        for src in range(1, self.world_size):
            thread = threading.Thread(
                target=self._handle_worker,
                args=(src,)
            )
            thread.daemon = True
            thread.start()
            self.recv_threads.append(thread)


    def _handle_worker(self, src):
        """处理指定GPU进程的通信"""
        notice = self.notices[src]
        new_tokens = 0
        while True:
            # 清空 recv_cache 和 update_cache
            notice.recv_cache.fill_(-1)
            notice.update_cache.fill_(-1)

            # 接收token数据
            token_ids = notice.recv_cache
            dist.recv(tensor=notice.recv_cache, src=src)
            prefix_len = self.get_truncate_input_ids_len(notice.recv_cache)
            print(f"线程{src}接受数据 {notice.recv_cache}, prefix_len is {prefix_len}")
            if src == 1:
                # smallest model, check notice to update
                with notice.lock:
                    if notice.is_update:
                        dist.send(tensor=notice.update_cache, dst=src)
                        notice.input_ids = notice.update_cache.clone()
                        notice.is_update = False
                        new_tokens = 0
                    else:
                        dist.send(tensor=notice.recv_cache, dst=src)
                        new_tokens += 1
                        notice.input_ids = notice.recv_cache.clone()
                        if new_tokens >= Config.PREDICTION_NUM:
                            with self.global_condition:
                                self.global_condition.notify_all()
            elif src == self.world_size - 1:
                # target model
                # 查询上一层model 的cache
                last_notice = self.notices[src - 1]
                with last_notice.lock:
                    cache_len = self.find_first_diff_index(last_notice.input_ids, notice.recv_cache)
                send_token_ids = last_notice.input_ids
                if cache_len < prefix_len:
                    # cache 未命中，通知其他模型更新
                    # 注意要从大到小进行通知，否则会出现线程安全问题
                    for i in range(self.world_size-1, 0, -1):
                        with self.notices[i].lock:
                            self.notices[i].is_update = True
                            self.notices[i].update_cache = notice.recv_cache.clone()
                    # Condition或事件代替忙等待，例如在更新input_ids后通知等待的线程。
                    with self.global_condition:
                        while True:
                            with self.notices[1].lock:
                                current_cache_len = self.find_first_diff_index(self.notices[1].input_ids, notice.recv_cache)
                            if current_cache_len == prefix_len:
                                break
                            self.global_condition.wait()
                    send_token_ids = self.notices[1].input_ids
                dist.send(tensor=send_token_ids, dst=src)

            else:
                # middle model, check notice to update
                # 查看自己是否需要更新
                if notice.is_update:
                    # 这里update 可能是 bigger model 发过来的
                    # todo在阶梯更高的模型层级中打开测试 取更小模型来加速推理，注意：这个一定要更小模型执行速度 为 当前模型2倍数以上，否则会出bug
                    # with self.notices[1].lock:
                        # if self.get_truncate_input_ids_len(self.notices[1].input_ids) > self.get_truncate_input_ids_len(notice.update_cache):
                        #     dist.send(tensor=self.notices[1].input_ids, dst=src)
                        # else:
                        #     dist.send(tensor=notice.update_cache, dst=src)
                    dist.send(tensor=notice.update_cache, dst=src)
                    notice.input_ids = notice.update_cache.clone()
                    notice.is_update = False
                    continue

                # 查询上一层model 的cache
                last_notice = self.notices[src - 1]
                cache_len = self.find_first_diff_index(last_notice.input_ids, notice.recv_cache)
                if cache_len < prefix_len:
                    # cache 未命中，通知其他模型更新
                    dist.send(tensor=notice.recv_cache, dst=src)
                    notice.input_ids = notice.recv_cache.clone()
                    with notice.lock:
                        if notice.is_update is not True:
                            for i in range(src-1, 0, -1):
                                with self.notices[i].lock:
                                    # todo target model 有类似的这种检查吗？
                                    if self.notices[i].is_update is not True and not torch.equal(self.notices[i].input_ids, notice.recv_cache):
                                        self.notices[i].is_update = True
                                        self.notices[i].update_cache = notice.recv_cache.clone()
                else:
                    # 命中
                    dist.send(tensor=last_notice.input_ids, dst=src)
                    notice.input_ids = last_notice.input_ids.clone()


    def get_truncate_input_ids_len(self, tensor: torch.Tensor) -> int:
        row = tensor[0]
        # 方法 2：使用 nonzero 查找位置
        indices = (row == -1).nonzero(as_tuple=True)[0]
        index = indices[0] if len(indices) > 0 else row.size(0)
        return index.item()

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