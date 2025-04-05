from time import perf_counter

import torch
import threading
import torch.distributed as dist
from src.Config import Config
from src.Notice import Notice
from typing import List


class CacheManager:
    def __init__(self, world_size, max_len = Config.BUFFER_SIZE):
        self.world_size = world_size
        self.lock = threading.Lock()
        self.recv_threads = []
        # in fact , we only need ${world_size} notice, but for easier indexing, we use world_size+1
        self.max_len: int = max_len
        self.notices: List[Notice] = [Notice(max_len) for i in range(world_size)]
        self.global_condition = threading.Condition()  # 全局条件变量
        self.terminate_flag: list = list()
        self.sum_time = 0

    def reset_cache_manager(self) -> None:
        """用于重置 Cache manager"""
        for src in range(1, self.world_size):
            self.notices[src].reset_notice()
            print(f"重置第{src}个cache manager 成功")
            print(f"update cache is {self.notices[src].update_cache}")

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

    def join(self):
        for e in self.recv_threads:
            e.join()

    def _handle_worker(self, src):
        """处理指定GPU进程的通信"""
        notice = self.notices[src]
        new_tokens = 0
        while src not in self.terminate_flag:
            # 清空 recv_cache 和 update_cache
            notice.recv_cache.fill_(-1)
            # notice.update_cache.fill_(-1)

            # 接收token数据
            token_ids = notice.recv_cache
            work = dist.irecv(tensor=notice.recv_cache, src=src)
            # 在这里对 update cache 进行健壮性检查
            with notice.lock:
                pass

            # notice.is_update = False
            try:
                work.wait()
            except RuntimeError as e:
                print(f"通信通道关闭,线程{src}结束")
                break
            prefix_len = self.get_truncate_input_ids_len(notice.recv_cache)
            if src == 1:
                # smallest model, check notice to update
                with notice.lock:
                    if self.find_first_diff_index(notice.recv_cache,notice.update_cache) == self.get_truncate_input_ids_len(notice.update_cache):
                        notice.is_update = False
                        notice.update_cache.fill_(-1)

                    if notice.is_update:
                        print(f"len(notice.update_cache) is {self.get_truncate_input_ids_len(notice.update_cache)}")
                        # print(f"self.find_first_diff_index(notice.recv_cache, notice.update_cache) is {self.find_first_diff_index(notice.recv_cache, notice.update_cache)}")

                        dist.send(tensor=notice.update_cache, dst=src)
                        notice.input_ids = notice.update_cache.clone()
                        notice.is_update = False
                        notice.update_cache.fill_(-1)
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
                    first_diff_index = self.find_first_diff_index(last_notice.input_ids, notice.recv_cache)
                send_token_ids = last_notice.input_ids
                if first_diff_index < prefix_len:
                    # cache 未命中，通知其他模型更新
                    # 注意要从大到小进行通知，否则会出现线程安全问题
                    for i in range(self.world_size-1, 0, -1):
                        with self.notices[i].lock:
                            self.notices[i].is_update = True
                            self.notices[i].update_cache = notice.recv_cache.clone()

                    if not Config.IS_BRANCH_PREDICTION: # 是否采用分支预测
                        dist.send(tensor=notice.recv_cache, dst=src)
                        continue
                    else:
                        # Condition或事件代替忙等待，例如在更新input_ids后通知等待的线程。
                        with self.global_condition:
                            self.global_condition.wait()
                        send_token_ids = self.notices[1].input_ids
                dist.send(tensor=send_token_ids, dst=src)

            else:
                # middle model, check notice to update
                # 查看自己是否需要更新
                with notice.lock:
                    if self.find_first_diff_index(notice.recv_cache,notice.update_cache) == self.get_truncate_input_ids_len(notice.update_cache):
                        notice.is_update = False
                        notice.update_cache.fill_(-1)
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
                        notice.update_cache.fill_(-1)
                        continue

                # 查询上一层model 的cache
                last_notice = self.notices[src - 1]
                first_diff_index = self.find_first_diff_index(last_notice.input_ids, notice.recv_cache)
                if first_diff_index < prefix_len:
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
        print(f"{src} cache manager 结束")

    def get_truncate_input_ids_len(self, tensor: torch.Tensor) -> int:
        row = tensor[0]
        indices = (row == -1).nonzero(as_tuple=True)[0]
        # 返回值统一处理为 Python int
        if len(indices) > 0:
            return indices[0].item()  # Tensor 转 int
        else:
            return row.size(0)  # 直接返回 int

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