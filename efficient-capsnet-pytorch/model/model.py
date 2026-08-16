import numpy as np
import torch
import torch.nn as nn

from .layers import CapsLen, CapsMask, PrimaryCaps, RoutingCaps


class EfficientCapsNet(nn.Module):
    def __init__(self, input_size=(3, 32, 32), num_classes=10, capsule_dim=16):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_size[0], out_channels=32, kernel_size=5, padding=0
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3)  # padding=0 is default
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCaps(
            in_channels=128, kernel_size=11, capsule_size=(16, 8)
        )
        self.routing_caps = RoutingCaps(in_capsules=(16, 8), out_capsules=(num_classes, capsule_dim))
        self.len_final_caps = CapsLen()
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.primary_caps(x)
        x = self.routing_caps(x)
        return x, self.len_final_caps(x)


class ReconstructionNet(nn.Module):
    def __init__(self, input_size=(3, 32, 32), num_classes=10, num_capsules=16):
        super(ReconstructionNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=num_capsules * num_classes, out_features=512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, np.prod(input_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.xavier_normal_(self.fc3.weight)  # glorot normal

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.view(-1, *self.input_size)  # reshape


class FinalCapsNet(nn.Module):
    def __init__(
        self,
        input_size=(3, 32, 32),
        num_classes=10,
        capsule_dim=16,
        use_background_class=False
    ):
        super(FinalCapsNet, self).__init__()

        self.num_classes = num_classes
        self.use_background_class = use_background_class

        n_out_caps = num_classes + 1 if use_background_class else num_classes

        self.efficient_capsnet = EfficientCapsNet(input_size, n_out_caps, capsule_dim)

        self.mask = CapsMask()
        self.generator = ReconstructionNet(input_size, n_out_caps, capsule_dim)

    def forward(self, x, y_true=None, mode='train'):
        x, x_len = self.efficient_capsnet(x)
        if mode == "train":
            if self.use_background_class:
                # background capsule is never a training target -- pad the
                # one-hot label with a 0 so shapes line up for masking
                bg_col = torch.zeros_like(y_true[:, :1])
                y_mask = torch.cat([y_true, bg_col], dim=1)
            else:
                y_mask = y_true
            masked = self.mask(x, y_mask)
        elif mode == "eval":
            masked = self.mask(x)
        x = self.generator(masked)
        digit_len = x_len[:, : self.num_classes]
        return x, digit_len
