from .cnn import *
from .gbsnet import *

from torchvision.models import *
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1


def gbs_lenet(hidden_size, n_a, num_layer=3, num_classes=10):
    backbone = ConvNet(hidden_size, n_a)
    return_layer = 'layer2'
    classifier = GbsCls(5*5*64, hidden_size, num_layer, n_a, num_classes)
    return gbs_conv(backbone, return_layer, classifier)


def gbs_squeeze1_1(hidden_size, n_a, num_layer=3, num_classes=10):
    backbone = squeezenet1_1
    return_layer = 'features'
    classifier = GbsCls(512, hidden_size, num_layer, n_a, num_classes)
    return gbs_conv(backbone, return_layer, classifier)


def _get_model(model_name, hidden_size, n_a, num_layer=3, num_classes=10):
    model_dict = {'lenet': gbs_lenet,
                  'squeezenet1_1': gbs_squeeze1_1}
    return model_dict[model_name](hidden_size, n_a, num_layer, num_classes)
