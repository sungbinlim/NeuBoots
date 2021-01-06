import os
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


def run(str):
    os.system(str)
    return 0


def main():
    strs = [f"python -m torch.distributed.launch --nproc_per_node=1 main.py deep --gpus {i%8} --seed {i}" for i in range(40)]
    processes = []
    for i in range(5):
        for rank in range(8):
            p = mp.Process(target=run, args=(strs[i * 8 + rank],))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
