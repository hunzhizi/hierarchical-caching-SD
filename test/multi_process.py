import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import threading
import time
from collections import defaultdict


def init_process(rank, world_size):
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:12345',
        rank=rank,
        world_size=world_size
    )
    if rank != 0:
        torch.cuda.set_device(rank - 1)  # 确保实际存在对应GPU


class GlobalTableManager:
    def __init__(self, world_size):
        self.world_size = world_size
        self.table = defaultdict(int)
        self.lock = threading.Lock()
        self.recv_threads = []

    def start(self):
        for src in range(1, self.world_size):
            thread = threading.Thread(
                target=self._handle_worker,
                args=(src,)
            )
            thread.daemon = True
            thread.start()
            self.recv_threads.append(thread)

    def _handle_worker(self, src):
        while True:
            data = torch.zeros(1024, dtype=torch.long)
            dist.recv(tensor=data, src=src)
            tokens = data[data > 0].tolist()

            with self.lock:
                for token in tokens:
                    self.table[token] += 1

                return_data = torch.tensor(
                    list(self.table.keys())[:1024],
                    dtype=torch.long
                )

            dist.send(tensor=return_data, dst=src)


def cpu_main(world_size):
    manager = GlobalTableManager(world_size)
    manager.start()
    while True:
        time.sleep(1)
        print(f"[CPU] Current table size: {len(manager.table)}")


def gpu_main(rank, world_size):
    device = torch.device(f'cuda:{rank - 1}')
    model = torch.nn.Linear(10, 10).to(device)

    # 修正：缓冲区置于CPU
    front_buffer = torch.zeros(1024, dtype=torch.long, device='cpu')
    back_buffer = torch.zeros(1024, dtype=torch.long, device='cpu')

    step = 0
    while True:
        # 修正：生成1024长度的假数据
        fake_output = torch.randint(100, 1000, (1024,), device=device)
        send_data = fake_output.cpu()

        dist.send(tensor=send_data, dst=0)
        req = dist.irecv(tensor=back_buffer)

        # 模拟计算与通信重叠
        # ...

        req.wait()

        front_buffer, back_buffer = back_buffer, front_buffer

        if step % 50 == 0:
            print(f"Rank {rank} step {step} received {len(front_buffer[front_buffer > 0])} tokens")
        step += 1


def main(rank, world_size):
    init_process(rank, world_size)
    if rank == 0:
        cpu_main(world_size)
    else:
        gpu_main(rank, world_size)


if __name__ == "__main__":
    world_size = 4  # 1 CPU + 3 GPU，确保实际有3块GPU
    mp.spawn(main, args=(world_size,), nprocs=world_size)