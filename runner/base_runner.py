import torch
from pathlib import Path
import shutil

from utils.logger import Logger


class BaseRunner(object):
    def __init__(self, args, loader, model):
        self.save_path = f"outs/{args.inifile}"
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(args, self.save_path)
        self.loader = loader
        self.G = model
        self.load()

    def load(self):
        pass
