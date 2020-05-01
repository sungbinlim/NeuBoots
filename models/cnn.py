import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, hidden_size, n_a, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(5*5*64 + n_a, hidden_size)
        self.Lrelu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(hidden_size + n_a, hidden_size)
        self.fc3 = nn.Linear(hidden_size + n_a, hidden_size)
        self.fc4 = nn.Linear(hidden_size + n_a, hidden_size)
        self.fc5 = nn.Linear(hidden_size + n_a, hidden_size)
        self.fc_out = nn.Linear(hidden_size + n_a, num_classes)
        self.soft = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(5*5*64)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

    def forward(self, x, w, fac1=5.0):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.bn1(out)
        out2 = fac1 * torch.exp(-1.0*w)
        out0 = torch.cat([out2, out], dim=1)
        out0 = self.Lrelu(self.fc1(out0))
        out0 = self.bn2(out0)
        out0 = torch.cat([out2, out0], dim=1)
        out0 = self.Lrelu(self.fc2(out0))
        out0 = self.bn3(out0)
        out0 = torch.cat([out2, out0], dim=1)
        out0 = self.Lrelu(self.fc3(out0))
        out0 = self.bn4(out0)
        out0 = torch.cat([out2,  out0], dim=1)
        out0 = self.fc_out(out0)
        out0 = self.soft(out0)
        return out0
