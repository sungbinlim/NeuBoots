import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from collections import OrderedDict


class LinearActBn(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.fc = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_feat, out_feat)),
            ('act', nn.LeakyReLU(inplace=False)),
            ('norm', nn.BatchNorm1d(out_feat))
        ]))

    def forward(self, x):
        return self.fc(x)


class GbsCls(nn.Module):
    def __init__(self, in_feat, hidden_size, num_layer, n_a, num_classes):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        for i in range(num_layer):
            fc = LinearActBn(in_feat + n_a, hidden_size)
            self.fc_layers.append(fc)
            in_feat = hidden_size
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x, w, fac1):
        out = x
        out2 = fac1 * torch.exp(-1.0*w)
        for i, layer in enumerate(self.fc_layers):
            # print(out.shape, out2.shape)
            out = layer(torch.cat([out, out2], dim=1))
        return self.fc_out(out).softmax(1)


class GbsConvNet(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifer = classifier

    def forward(self, x, w, fac1):
        feat = self.backbone(x)
        out = feat['out']
        if out.size(-1) != 1:
            out = out.view(out.size(0), -1)
        else:
            out = out.squeeze()
        return self.classifer(out, w, fac1)


def gbs_conv(backbone, return_layer, classifier):
    layer_dict = {return_layer: 'out'}
    backbone = IntermediateLayerGetter(backbone(), layer_dict)
    model = GbsConvNet(backbone, classifier)
    return model


def D(Prob, y1, w1):
    out = -1.0 * y1 * torch.log(Prob)
    out = out.sum(1).view([-1, 1])
    out = out * w1
    return out.sum()
