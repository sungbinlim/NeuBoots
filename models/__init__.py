from .cnn import lenet
from .gbsnet import gbs_conv, GbsCls

from torchvision import models


MODEL_DICT = {'lenet': [lenet, 'layer2', 5*5*64],
              'alexnet': [models.alexnet, 'avgpool', 256*6*6],
              'resnet50': [models.resnet50, 'avgpool', 2048],
              'squeeze1_0': [models.squeezenet1_0, 'features', 512],
              'mnasnet0_5': [models.mnasnet0_5, 'layers', 1280]}


def _get_model(model_name, hidden_size, n_a, num_layer, num_classes):
    backbone, return_layer, in_feat = MODEL_DICT[model_name]
    classifier = GbsCls(in_feat, hidden_size, num_layer, n_a, num_classes)
    return gbs_conv(backbone, return_layer, classifier)
