import numpy as np
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw


def get_transform(crop_size, padding, cutout_size, data_type='cifar10'):
    if data_type == 'stl':
        mean = (0.4409, 0.4279, 0.3868)
        std = (0.2683, 0.2610, 0.2687)
    elif data_type == 'voc':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    train_list = []
    test_list = []
    if data_type == 'voc':
        train_list += [
            PairResize(520),
            PairRandomCrop(crop_size, padding=padding),
            PairRandomHorizontalFlip(),
            PairToTensor(),
            PairNormalize(mean, std)
        ]
        test_list += [
            PairResize(520),
            PairCenterCrop(crop_size),
            PairToTensor(),
            PairNormalize(mean, std)
        ]
        transform_train = PairCompose(train_list)
        transform_test = PairCompose(test_list)
    else:
        train_list += [
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        test_list += [transforms.ToTensor(),
                    transforms.Normalize(mean, std)]

        transform_train = transforms.Compose(train_list)
        transform_test = transforms.Compose(test_list)

    return {'train': transform_train, 'test': transform_test}


class PairCompose(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, img, label):
        # label = label.convert('L')
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class PairCenterCrop(transforms.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def __call__(self, img, label):
        return F.center_crop(img, self.size), F.center_crop(label, self.size)


class PairResize(transforms.Resize):
    def __init__(self, size):
        super().__init__(size)

    def __call__(self, img, label):
        return F.resize(img, self.size, self.interpolation), F.resize(label, self.size)


class PairRandomCrop(transforms.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def __call__(self, img, label):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - label.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)


class PairRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, img, label):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, img, label):
        label = np.array(label)
        label[label == 255] = 0
        return F.to_tensor(img), torch.from_numpy(label).long()


class PairNormalize(transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace)

    def __call__(self, img, label):
        return F.normalize(img, self.mean, self.std, self.inplace), label


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MnistNorm(object):
    def __call__(self, img):
        img = F.to_tensor(img).view(-1)#.repeat_interleave(3, dim=0)
        return img


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img
