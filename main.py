import argparse

from utils.logger import get_logger
from utils.arg_parser import Argments
from loader.data_loader import GeneralDataLoader, NbsDataLoader
from runners.cnn_runner import CnnRunner
from runners.nbs_runner import NbsRunner


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("yaml")
    argparser.add_argument("--phase", default="train", type=str)
    argparser.add_argument("--seed", default=-1, type=int)
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
    if model_type == 'nbs':
        data_loader = NbsDataLoader(dataset, setup['batch_size'],
                                    setup['n_a'], setup['cpus'], setup['seed'])
        runner = NbsRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                           logger=logger, model_path=model_path, rank=setup['rank'],
                           epoch_th=setup['epoch_th'], num_mc=setup['num_mc'],
                           adv_training=setup['adv_training'])
    else:
        data_loader = GeneralDataLoader(dataset, setup['batch_size'],
                                        setup['cpus'], setup['seed'])
        runner = CnnRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                           logger=logger, model_path=model_path, rank=setup['rank'],
                           adv_training=setup['adv_training'])

    if setup['phase'] == 'train':
        runner.train()
        runner.test()
    elif setup['phase'] == 'test':
        runner.test()


if __name__ == "__main__":
    main()
