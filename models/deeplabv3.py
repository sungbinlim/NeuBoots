import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabHead


class GeneralDeepLabHead(DeepLabHead):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
        self.num_classes = num_classes


class NbsDeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.input_shape = 513
        self.feature = nn.Sequential(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, alpha):
        x = self.feature(x)
        if isinstance(alpha, int):
            res_ = torch.zeros([alpha, x.size(0), self.num_classes,
                            self.input_shape, self.input_shape]).cuda()
            for i in range(alpha):
                w = torch.rand([x.size(0), 256, 1, 1]).cuda()
                res = self.classifier(x * w)
                res_[i] += F.interpolate(res, self.input_shape,
                                mode='bilinear', align_corners=False)
            return res_
        else:
            w = torch.exp(-F.interpolate(alpha[:, None], 256))[:, 0, :, None, None]
            out = self.classifier(x * w)
            return out
