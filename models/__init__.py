import torch

from torchvision import models

from .gbsnet import gbs_conv, GbsCls
from .densenet import densenet100
from .wideresnet import wresnet28_2, wresnet28_10, wresnet16_8


MODEL_DICT = {'alexnet': [models.alexnet, 'avgpool', 256 * 6 * 6],
              'resnet34': [models.resnet34, 'avgpool', 512],
              'resnet50': [models.resnet50, 'avgpool', 2048],
              'densenet100': [densenet100, 'bn1', 342],
              'squeeze1_0': [models.squeezenet1_0, 'features', 512],
              'mnasnet0_5': [models.mnasnet0_5, 'layers', 1280],
              'wresnet28_2': [wresnet28_2, 'avgpool', 128],
              'wresnet16_8': [wresnet16_8, 'avgpool', 512],
              'wresnet28_10': [wresnet28_10, 'avgpool', 640]}


def _get_model(model_name, hidden_size, n_a, num_layer,
               num_classes, is_gbs=True, dropout_rate=0.):
    backbone, return_layer, in_feat = MODEL_DICT[model_name]
    if is_gbs:
        classifier = GbsCls(in_feat, hidden_size, num_layer, n_a, num_classes)
    else:
        classifier = torch.nn.Linear(in_feat, num_classes)
    return gbs_conv(backbone, return_layer, classifier, is_gbs, dropout_rate)
