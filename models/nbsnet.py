import math
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict


class NbsCls(nn.Module):
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self.fc_out = nn.Linear(in_feat, num_classes)
        self.num_classes = num_classes

    def forward(self, x, alpha):
        out1 = x
        if isinstance(alpha, int):
            res_ = torch.zeros([alpha, out1.size(0), self.num_classes]).cuda()
            for i in range(alpha):
                w = torch.rand_like(out1).cuda()
                res = self.fc_out(out1 * w)
                res_[i] += res
            return res_
        else:
            out2 = torch.exp(-F.interpolate(alpha[:, None], self.in_feat))[:, 0]
            return self.fc_out(out1 * out2)


class NbsConvNet(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifer = classifier

    def forward(self, x, w=None):
        out = self.backbone(x)
        if out.size(-1) != 1:
            out = F.relu(out, inplace=True).mean([2, 3])
        else:
            out = out.squeeze()
        return self.classifer(out, w)


class GeneralConvNet(nn.Module):
    def __init__(self, backbone, classifier, last_drop=.0):
        super().__init__()
        self.backbone = backbone
        self.fc = classifier
        self.dropout = nn.Dropout(p=last_drop)

    def forward(self, x):
        out = self.backbone(x)
        if out.size(-1) != 1:
            out = F.relu(out, inplace=True).mean([2, 3])
        else:
            out = out.squeeze()
        out = self.dropout(out)
        return self.fc(out)


class BackboneGetter(nn.Sequential):
    def __init__(self, model, return_layer):
        if not set([return_layer]).issubset([name for name, _ in
                                             model.named_children()]):
            raise ValueError("return_layer is not present in model")

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name == return_layer:
                break

        super().__init__(layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


def get_conv(backbone, return_layer, classifier, model_type, drop_rate=0.0):
    if model_type == 'mcd':
        backbone = BackboneGetter(backbone(drop_rate), return_layer)
    else:
        backbone = BackboneGetter(backbone(.0), return_layer)
    if model_type == 'nbs':
        model = NbsConvNet(backbone, classifier)
    else:
        model = GeneralConvNet(backbone, classifier, drop_rate)
    return model
