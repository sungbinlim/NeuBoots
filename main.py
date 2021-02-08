import argparse

from utils.logger import get_logger
from utils.arg_parser import Argments
from loader.data_loader import GeneralDataLoaderCls, NbsDataLoaderCls,\
                               GeneralDataLoaderSeg, NbsDataLoaderSeg
from runners.cnn_runner import CnnRunner
from runners.nbs_runner import NbsRunner
from runners.mcd_runner import McdRunner


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("yaml")
    argparser.add_argument("--phase", default="train", type=str)
    argparser.add_argument("--index", default=-1, type=int)
    argparser.add_argument("--gpus", default="-1", type=str)
    argparser.add_argument("--local_rank", default=0, type=int)
    cmd_args = argparser.parse_args()

    arg = Argments(f"scripts/{cmd_args.yaml}.yaml", cmd_args)
    setup = arg['setup']
    model_path = arg['path/model_path']
    logger = get_logger(f"{model_path}/log.txt")

    if setup['rank'] == 0:
        logger.info(arg)

    model_type = setup['model_type']
    dataset = arg['path/dataset']
    is_seg = False
    if 'nbs' in model_type:
        if 'seg' in model_type:
            is_seg = True
            _data_loader = NbsDataLoaderSeg
        else:
            _data_loader = NbsDataLoaderCls
        data_loader = _data_loader(dataset, setup['batch_size'],
                                   setup['n_a'], setup['cpus'], setup['seed'])
        runner = NbsRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                           logger=logger, model_path=model_path, rank=setup['rank'],
                           epoch_th=setup['epoch_th'], num_mc=setup['num_mc'],
                           adv_training=setup['adv_training'])
    else:
        if 'seg' in model_type:
            is_seg = True
            _data_loader = GeneralDataLoaderSeg
        else:
            _data_loader = GeneralDataLoaderCls
        data_loader = _data_loader(dataset, setup['batch_size'],
                                   setup['cpus'], setup['seed'])
        if 'mcd' in model_type:
            runner = McdRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                               logger=logger, model_path=model_path, rank=setup['rank'],
                               num_mc=setup['num_mc'], adv_training=setup['adv_training'])
        else:
            runner = CnnRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                            logger=logger, model_path=model_path, rank=setup['rank'],
                            adv_training=setup['adv_training'])

    if setup['phase'] == 'train':
        runner.train()
        runner.test(is_seg)
    elif setup['phase'] == 'test':
        runner.test(is_seg)


if __name__ == "__main__":
    main()
