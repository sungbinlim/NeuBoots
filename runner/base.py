import torch
from pathlib import Path
import shutil


class BaseRunner(object):
    def __init__(self, loader, inifile, num_epoch, model, optim, lr_schdlr):
        self.save_path = f"outs/{inifile}"
        self.loader = loader
        self.num_epoch = num_epoch
        self.epoch = 0
        self.G = model
        self.optim = optim
        self.lr_schler = lr_schdlr
        self.best_metric = 0
        self.load()

    def save(self, epoch, metric, file_name="model"):
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({"epoch": epoch,
                    "param": self.G.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    "best": self.best_metric,
                    "lr_schdlr": self.lr_schler.state_dict()
                    }, f"{save_path}/{file_name}.pth")

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

            self.G.load_state_dict(ckpoint['param'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.epoch = ckpoint['epoch']
            self.best_metric = ckpoint["score"]
            self.lr_schler.load_state_dict(ckpoint['lr_schdlr'])
            print(f"Load Model Type : {file_name}, epoch : {self.epoch}")
        else:
            print("Load Failed, not exists file")

    def get_lr(self):
        return self.lr_schler.optimizer.param_groups[0]['lr']
