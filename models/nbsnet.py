import math
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict


class NbsCls(nn.Module):
    def __init__(self, in_feat, n_a, num_classes, feature_adaptive=True):
        super().__init__()
        self.in_feat = in_feat
        if feature_adaptive:
            self.fc_out = nn.Linear(in_feat, num_classes)
        else:
            self.fc_out = nn.Linear(in_feat * 2, num_classes)
        self.n_a = n_a
        self.feature_adaptive = feature_adaptive
        self.num_classes = num_classes

    def forward(self, x, alpha):
        out1 = x
        if isinstance(alpha, int):
            res_ = torch.zeros([alpha, out1.size(0), self.num_classes]).cuda()
            for i in range(alpha):
                w = torch.rand_like(out1).cuda()
                if self.feature_adaptive:
                    res = self.fc_out(out1 * w)
                else:
                    res = self.fc_out(torch.cat([out1, w], dim=1))
                res_[i] += res
            return res_
        else:
            if self.in_feat != self.n_a:
                out2 = torch.exp(-F.interpolate(alpha[:, None],
                                 self.in_feat))[:, 0]
            else:
                out2 = torch.exp(-alpha)
            if self.feature_adaptive:
                return self.fc_out(out1 * out2)
            else:
                return self.fc_out(torch.cat([out1, out2], dim=1))


class NbsConvNet(nn.Module):
    def __init__(self, backbone, classifier, is_nbs):
        super().__init__()
        self.backbone = backbone
        self.classifer = classifier
        self.is_nbs = is_nbs

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, w=None):
        out = self.backbone(x)
        if out.size(-1) != 1:
            out = F.relu(out, inplace=True).mean([2, 3])
        else:
            out = out.squeeze()
        if self.is_nbs:
            return self.classifer(out, w)
        else:
            return self.classifer(out)


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


def nbs_conv(backbone, return_layer, classifier, is_nbs, dropout_rate):
    backbone = BackboneGetter(backbone(dropout_rate), return_layer)
    model = NbsConvNet(backbone, classifier, is_nbs)
    return model


def D(Prob, y1, w1=None, reduce='mean'):
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    out = ce(Prob, y1)
    if w1 is None:
        return out.mean()
    out = out * w1
    if reduce == 'mean':
        return out.mean()
    else:
        return out.sum()
