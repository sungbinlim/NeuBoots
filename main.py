import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler, Adam, RMSprop, SGD
import torch.distributed as dist

import os
import argparse

from data.mnist_loader import MnistLoader
from data.cifar_loader import Cifar10Loader
from runner.gbs_runner import GbsCnnClsfier
from models.gbsnet import D
from models import _get_model

from utils.arg_parser import parse_args


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("inifile")
    argparser.add_argument("--local_rank", default=0, type=int)
    cmd_args = argparser.parse_args()

    args = parse_args(cmd_args)

    args.distributed = False
    if args.dist:
        if 'WORLD_SIZE' in os.environ:
            args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.distributed:
        torch.cuda.set_device(cmd_args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    # data_loader = MnistLoader(args.batch_size, args.n_a, args.sub_size,
    data_loader = Cifar10Loader(args.batch_size, args.n_a, args.sub_size,
                                args.cpus, args.v)
    p = data_loader.p
    model, optim = get_model_optim(args, p)
    lr_schdlr = lr_scheduler.CyclicLR(optim, base_lr=args.lr,
                                      max_lr=args.lr_max,
                                      step_size_up=1000)
    loss_fn = D

    runner = GbsCnnClsfier(args, data_loader, model, optim, lr_schdlr, loss_fn)
    if args.phase == 'train':
        runner.train()
        runner.test()
    else:
        runner.test()


def get_model_optim(args, p):
    hidden_size = p if p >= 100 else 100
    model = _get_model(args.model, hidden_size, args.n_a,
                       args.num_layer, args.num_classes).cuda()
    if args.optim == 'adam':
        Optim = Adam
    elif args.optim == 'rmsp':
        Optim = RMSprop
    elif args.optim == 'sgd':
        Optim = SGD
    optim = Optim(model.parameters(), lr=args.lr,
                  weight_decay=args.weight_decay)
                #   nesterov=True, momentum=0.9)

    if args.dist:
        if args.apex:
            from apex import parallel, amp
            model, optim = amp.initialize(model, optim,
                                          opt_level=args.amp_level)
            Parallel = parallel.DistributedDataParallel
        else:
            Parallel = DistributedDataParallel
    else:
        Parallel = DataParallel

    return Parallel(model), optim


if __name__ == "__main__":
    main()
