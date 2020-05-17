import math
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
        in_feat += n_a
        for i in range(num_layer):
            fc = LinearActBn(in_feat, hidden_size)
            self.fc_layers.append(fc)
            in_feat = hidden_size + n_a
        self.fc_out = nn.Linear(in_feat, num_classes)

    def forward(self, x, w, fac1):
        out = x
        out2 = fac1 * torch.exp(-1.0 * w)
        for i, layer in enumerate(self.fc_layers):
            out = layer(torch.cat([out, out2], dim=1))
        return self.fc_out(torch.cat([out, out2], dim=1))


class GbsConvNet(nn.Module):
    def __init__(self, backbone, classifier, is_gbs):
        super().__init__()
        self.backbone = backbone
        self.classifer = classifier
        self.is_gbs = is_gbs

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, w, fac1):
        feat = self.backbone(x)
        out = feat['out']
        if out.size(-1) != 1:
            out = out.view(out.size(0), -1)
        else:
            out = out.squeeze()
        if self.is_gbs:
            return self.classifer(out, w, fac1)
        else:
            return self.classifer(out)


def gbs_conv(backbone, return_layer, classifier, is_gbs):
    layer_dict = {return_layer: 'out'}
    backbone = IntermediateLayerGetter(backbone(), layer_dict)
    model = GbsConvNet(backbone, classifier, is_gbs)
    return model


def D(Prob, y1, w1=None, reduce='sum'):
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    out = ce(Prob, y1)
    if not w1:
        return out.mean()
    out = out[..., None] * w1
    if reduce == 'mean':
        return out.mean()
    else:
        return out.sum()
