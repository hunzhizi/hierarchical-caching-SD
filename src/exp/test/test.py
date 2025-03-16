import argparse
from time import sleep
from random import randint
from torch.multiprocessing import Process
import torch.distributed as dist
import torch



def initialize(rank, world_size):
    dist.init_process_group(backend='gloo',
                            init_method='file://D:/tmp',
                            rank=rank, world_size=world_size)
    if rank == 0:
        dist.send(tensor=torch.tensor([randint(0, 10) for _ in range(5)], dtype=torch.float32), dst=1)
    else:
        recv_data = torch.zeros(5, dtype=torch.float32)
        dist.recv(tensor=recv_data, src=0)
        print(recv_data)



def main():
    size = 2
    processes = []
    for i in range(size):
        p = Process(target=initialize, args=(i, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()