import datetime
import random
import time
from time import sleep

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.multiprocessing import Process

from src.Config import Config

os.environ["GLOO_SOCKET_IFNAME"] = "WLAN"

from src.CacheManager import CacheManager
# 在代码中添加地址验证
  # 确保输出与init_method中的IP一致
# loaded_input_ids:torch.Tensor = torch.load("D:\\homework\\hierarchical-caching-SD\\data\\input_ids.pt")
tensor_1d = torch.arange(1, 2049)

# 将一维张量扩展为二维张量，形状为 (1, 2048)
loaded_input_ids = tensor_1d.unsqueeze(0)
random.seed(1234)

def cpu_main(world_size):
    """中央控制进程（rank 0）"""
    manager = CacheManager(world_size)
    manager.start()

    # 保持主线程存活
    while True:
        time.sleep(1)
        # 可在此处添加监控逻辑


def gpu_main(rank, world_size):
    """GPU工作进程（rank > 0）"""
    # 初始化设备
    device = torch.device(f'cpu')

    # 初始化模型（示例）
    # model = torch.nn.Linear(10, 10).to(device)

    # 双缓冲机制
    buffer = torch.full((1, Config.BUFFER_SIZE), -1, dtype=torch.long)
    step = 1
    seq_len = 1
    fake_output = torch.zeros(Config.BUFFER_SIZE, dtype=torch.long).unsqueeze(0)
    fake_output.fill_(-1)
    while True:
        # 使用前缓冲区数据进行推理

        # prefill 过程
        if step == 1:
            fake_output[:, 0] = loaded_input_ids[:, 0]
            step += 1
            send_data = fake_output.cpu()
            sleep(rank * rank * 0.0001)
            continue

        # decode
        if step != 1:
            dst = 0
            # 异步发送到CPU进程 请求cache
            dist.send(tensor=send_data, dst=0)
            color_print(f"rank{rank}:发送数据 {send_data} 到 rank{dst}", rank)

            # 非阻塞接收更新数据
            dist.recv(tensor=buffer,src=0)
            color_print(f"rank{rank}:接收数据 {buffer}", rank)
            # 用接受的数据作为下一次的输入
            fake_output[:,:] = buffer[:,:]
            # 回滚
            seq_len = get_truncate_input_ids(buffer).shape[1]
            fake_output[:, seq_len:] = -1


        # 模拟推理所需要的时间
        sleep(rank * rank * 0.0001)

        # 模拟验证
        # 准备发送数据（拷贝到CPU）
        index = find_first_diff_index(buffer, loaded_input_ids)
        if rank == 2:
            for i in range(index - seq_len):
                if control_probability(0.2):
                    # 验证失败
                    fake_output[:, seq_len ] = random.randint(1, 151936)
                    break
                else:
                    fake_output[:, seq_len ] = buffer[:, seq_len]
                seq_len += 1
            # 模拟回滚
            fake_output[:, seq_len:] = -1
            # 模拟sample 出下一个元素
            fake_output[:, seq_len] =loaded_input_ids[:, seq_len] if control_probability(0.8)  else random.randint(1, 151936)
            seq_len += 1
        elif rank == 3:
            index = find_first_diff_index(buffer, loaded_input_ids)
            print(f'find_first_diff_index: {index}')
            seq_len = index
            # 模拟 sample 出一个 token
            seq_len += 1
            fake_output[:, :seq_len] = loaded_input_ids[:, :seq_len]
            # 模拟回滚，后面数据清零
            fake_output[:, seq_len:] = -1
        elif rank == 1:
            if control_probability(0.4):
                fake_output[:, seq_len ] = random.randint(1, 151936)
            else:
                fake_output[:, seq_len ] = loaded_input_ids[:, seq_len]
            seq_len += 1

        # if rank == 0 and control_probability(0.4):
        #     fake_output[:, step] = random.randint(1,151936)
        # elif rank == 1 and control_probability(0.2):
        #     fake_output[:, step] = random.randint(1, 151936)

        send_data = fake_output.cpu()
        # req = dist.irecv(tensor=back_buffer)
        step += 1

def color_print( content: str, color_number: int = 4):
    """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
    # if self.accelerator.is_main_process:
    print(f"\033[9{color_number}m{content}\033[0m")

def find_first_diff_index(a: torch.Tensor,
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
        first_diff = diff_mask.nonzero(as_tuple=True)[1].min()
        return first_diff.item()

    # 全部相同则返回最小长度
    return min_len


def get_truncate_input_ids(tensor: torch.Tensor):
    row = tensor[0]
    # 找到所有为-1的索引
    indices = (row == -1).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        index = indices[0]
        return row[:index].unsqueeze(0)
    else:
        return tensor


def control_probability(probability):
    """
    根据传入的概率大小返回一个布尔值。

    :param probability: 控制的概率，取值范围为 0 到 1 之间的浮点数
    :return: 如果生成的随机数小于概率值，返回 True；否则返回 False
    """
    # 检查概率值是否在有效范围内
    if not (0 <= probability <= 1):
        raise ValueError("概率值必须在 0 到 1 之间。")

    # 生成一个 0 到 1 之间的随机浮点数
    rand_num = random.random()

    # 根据随机数和概率值的比较结果返回布尔值
    return rand_num < probability

def init_process(rank, world_size):
    dist.init_process_group(
        backend='gloo',
        # Use corrected file path or TCP
        init_method='tcp://localhost:12345',
        # init_method='file://D:/tmp',
        world_size=world_size,
        rank=rank
    )

    print(f"Rank {rank} initialized")


def main(rank, world_size):
    init_process(rank, world_size)
    if rank == 0:
        cpu_main(world_size)
    else:
        gpu_main(rank, world_size)


if __name__ == "__main__":
    world_size = 4  # 1 CPU + 3 GPU
    mp.spawn(main, args=(world_size,), nprocs=world_size)
