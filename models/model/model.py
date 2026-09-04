import numpy as np
import timm
import torch
import torch.nn as nn

from .layers import CapsLen, CapsMask, PrimaryCaps, RoutingCaps


def _conv2d_out_size(size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """Spatial output size of a single Conv2d application along one dimension."""
    return (size + 2 * padding - kernel_size) // stride + 1


def _capsnet_stem_out_size(size: int) -> int:
    """Spatial size remaining after EfficientCapsNet's 4 conv layers, along one dim.

    Mirrors conv1 (k5,s1,p0) -> conv2 (k3,s1,p0) -> conv3 (k3,s1,p0) ->
    conv4 (k3,s2,p0). For the paper's 32x32 input this returns 11, which is
    exactly the PrimaryCaps kernel size originally hardcoded below -- so
    computing it dynamically here is a drop-in generalization that lets
    EfficientCapsNet/FinalCapsNet accept any input resolution (e.g. 32x32
    up to 254x254 for the Flame dataset) instead of only 32x32.
    """
    size = _conv2d_out_size(size, kernel_size=5)
    size = _conv2d_out_size(size, kernel_size=3)
    size = _conv2d_out_size(size, kernel_size=3)
    size = _conv2d_out_size(size, kernel_size=3, stride=2)
    return size


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

        # PrimaryCaps' depthwise conv must collapse the conv stem's output
        # feature map down to exactly 1x1 (its forward() reshapes straight
        # to (batch, num_capsules, dim_capsules) with no spatial dims left).
        # So its kernel size has to equal the stem's output spatial size,
        # which depends on input_size -- computed here instead of the
        # original hardcoded 11 (valid only for 32x32 input).
        h_out = _capsnet_stem_out_size(input_size[1])
        w_out = _capsnet_stem_out_size(input_size[2])
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"input_size {input_size} is too small for EfficientCapsNet's "
                "conv stem (needs roughly >=16x16 after the 4 conv layers)."
            )
        primary_kernel_size = h_out if h_out == w_out else (h_out, w_out)
        # Note: PrimaryCaps' depthwise conv kernel scales with input_size (e.g.
        # ~11x11 for 32x32 input vs. ~122x122 for 254x254 input). It's still a
        # valid conv (kernel == remaining feature map, output is 1x1), but the
        # per-sample compute/memory of that single depthwise layer grows
        # roughly with input_size^4, so training at native 254x254 is
        # substantially slower/heavier than at smaller configured sizes.

        self.primary_caps = PrimaryCaps(
            in_channels=128, kernel_size=primary_kernel_size, capsule_size=(16, 8)
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

class TimmClassifier(nn.Module):
    """Adapter that exposes any timm classification model through the same
    forward(x, y_true=None, mode='train') -> (recon, logits) interface the
    trainer expects. Optionally applies the standard CIFAR-scale
    adaptations for ResNets and ViT/DeiT models built for 224x224 inputs.
    """
    def __init__(
        self,
        model_name,
        num_classes=10,
        pretrained=False,
        cifar_stem=False,
        patch_size=None,
        img_size=None,
        **timm_kwargs,
    ):
        """
        Args:
            model_name: any timm model name, e.g. 'resnet18', 'deit_tiny_patch16_224'.
            num_classes: number of output classes.
            pretrained: load ImageNet weights (usually False when also
                changing input resolution/patch size -- shapes won't match).
            cifar_stem: if True, replaces a ResNet-style 7x7 stride-2 stem +
                maxpool with a 3x3 stride-1 conv and no maxpool, per the
                standard CIFAR adaptation (see e.g. He et al.-style repos).
                Only applies to models with a `.conv1`/`.maxpool` (ResNet family).
            patch_size: override ViT/DeiT patch size (e.g. 4 for 32x32 inputs
                instead of the default 16). Ignored for non-patch models.
            img_size: override the input resolution the model was built for
                (needed alongside patch_size so position embeddings are sized
                correctly). Ignored for non-patch models.
        """
        super().__init__()
        if patch_size is not None:
            timm_kwargs["patch_size"] = patch_size
        if img_size is not None:
            timm_kwargs["img_size"] = img_size

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, **timm_kwargs
        )

        if cifar_stem:
            self._apply_cifar_stem()

    def _apply_cifar_stem(self):
        if not hasattr(self.backbone, "conv1"):
            raise ValueError(
                "cifar_stem=True expects a ResNet-style model with a "
                "'.conv1' stem (e.g. 'resnet18', 'resnet34', 'resnet50')."
            )
        old_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=old_conv1.in_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        # replace the stride-2 maxpool with a no-op so 32x32 isn't
        # downsampled to 8x8 before the first residual block even runs
        self.backbone.maxpool = nn.Identity()

    def forward(self, x, y_true=None, mode='train'):
        logits = self.backbone(x)
        return x, logits  # ClassificationLoss ignores the echoed image
