import torch
from pathlib import Path
import shutil

from utils.logger import Logger


class BaseRunner(object):
    def __init__(self, args, loader, model, optim, lr_schdlr):
        self.save_path = f"outs/{args.inifile}"
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(args, self.save_path)
        self.loader = loader
        self.num_epoch = args.num_epoch
        self.epoch = 0
        self.G = model
        self.optim = optim
        self.lr_schdlr = lr_schdlr
        self.best_metric = 0
        self.load()

    def save(self, epoch, metric, file_name="model", **kwargs):
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({"epoch": epoch,
                    "param": self.G.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    "best": self.best_metric,
                    "lr_schdlr": self.lr_schdlr.state_dict(),
                    **kwargs}, f"{save_path}/{file_name}.pth")

        if metric >= self.best_metric:
            print(f"{self.best_metric} -------------------> {metric}")
            self.best_metric = metric
            shutil.copy2(f"{save_path}/{file_name}.pth",
                         f"{save_path}/best.pth")
            print(f"Model has saved at {epoch} epoch.")

    def load(self, file_name="model.pth"):
        save_path = Path(self.save_path)
        print(save_path)
        if (save_path / file_name).exists():
            print(f"Load {save_path} File")
            ckpoint = torch.load(f"{save_path}/{file_name}")

            for key, value in ckpoint.items():
                if key == 'param':
                    self.G.load_state_dict(value)
                elif key == 'optimizer':
                    self.optim.load_state_dict(value)
                elif key == 'lr_schdlr':
                    self.lr_schdlr.load_state_dict(value)
                elif key == 'epoch':
                    self.epoch = value
                elif key == 'score':
                    self.best_metric = value
                else:
                    self.__dict__[key] = value

            print(f"Load Model Type : {file_name}, epoch : {self.epoch}")
        else:
            print("Load Failed, not exists file")

    def get_lr(self):
        return self.lr_schdlr.optimizer.param_groups[0]['lr']
