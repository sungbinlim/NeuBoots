import os
from runner.active_runner import ActiveRunner
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", default='cifar10')
    argparser.add_argument("--type", default='nbs_fa')
    argparser.add_argument("--gpu", default='0')
    argparser.add_argument("--fold", default=0)
    cmd_args = argparser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpu

    ar = ActiveRunner(cmd_args.dataset, 'resnet18', 400, 2000, 200, cmd_args.type, cmd_args.fold)
    ar.train_coreset()
    for i in range(10):
        ar.train_next_query(f"query{i}")


if __name__ == "__main__":
    main()
