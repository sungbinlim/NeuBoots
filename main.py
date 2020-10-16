import os
import torch
import argparse

import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler, Adam, RMSprop, SGD

from models.nbsnet import D
from models import _get_model
from utils.arg_parser import parse_args
from data.data_loader import NbsDataLoader
from runner.nbs_runner import NbsCnnClsfier

torch.backends.cudnn.benchmark = True


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

    data_loader = NbsDataLoader(args.dataset, args.batch_size,
                                args.n_a, args.sub_size, args.cpus)
    p = data_loader.p
    model, optim = get_model_optim(args, p)
    length = len(data_loader.load('train'))
    if args.scheduler == 'cosine':
        lr_schdlr = lr_scheduler.CosineAnnealingLR(optim,
                                                   length * args.num_epoch,
                                                   0.)
    elif args.scheduler == 'step':
        steps = [60 * length, 120 * length, 150 * length, 180 * length]
        lr_schdlr = lr_scheduler.MultiStepLR(optim, steps, 0.2)
    loss_fn = D

    runner = NbsCnnClsfier(args, data_loader, model, optim, lr_schdlr, loss_fn)
    if args.phase == 'train':
        runner.train()
        runner.test()
    else:
        runner.test()


def get_model_optim(args, p):
    model = _get_model(args.model, args.n_a, args.num_classes,
                       args.is_nbs, args.dropout_rate).cuda()
    if args.optim == 'adam':
        Optim = Adam
    elif args.optim == 'rmsp':
        Optim = RMSprop
    elif args.optim == 'sgd':
        Optim = SGD
    optim = Optim(model.parameters(), lr=args.lr,
                  weight_decay=args.weight_decay,
                  nesterov=True, momentum=0.9)

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
