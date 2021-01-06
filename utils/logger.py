import logging
from pathlib import Path


def get_logger(save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(save_path)
    logger.setLevel(logging.DEBUG)
    stream_hdl = logging.StreamHandler()
    file_hdl = logging.FileHandler(save_path)
    file_hdl.setLevel(logging.INFO)
    logger.addHandler(stream_hdl)
    logger.addHandler(file_hdl)
    return logger
