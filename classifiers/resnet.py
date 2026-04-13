'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifiers.seeded_layers import SeededLinear, SeededConv2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, model_rng, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SeededConv2d(model_rng, in_planes, planes,
                                  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SeededConv2d(model_rng, planes, planes, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeededConv2d(model_rng, in_planes, self.expansion*planes,
                             kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, model_rng, num_classes=10, in_channels=3, dropout=None, add_head=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = SeededConv2d(model_rng, in_channels, 64, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(model_rng, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(model_rng, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(model_rng, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(model_rng, block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if add_head:
            self.linear = SeededLinear(model_rng, 512*block.expansion, num_classes)

    def _make_layer(self, model_rng, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(model_rng, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _encode(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        out = self.layer2(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        out = self.layer3(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        out = self.layer4(out)
        out = self.avgpool(out)
        if self.dropout is not None:
            out = F.dropout2d(out, p=self.dropout)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_freeze_backbone", False):
            # Keep BN layers in eval mode when backbone is frozen
            for name, module in self.named_modules():
                if name.startswith("linear"):
                    continue
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        return self

    def forward(self, x):
        out = self._encode(x)
        if hasattr(self, "linear"):
            out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def load_pretrained_backbone(model, pretrained_path):
    """Load backbone weights from a SimCLR/ContrastiveModel checkpoint or raw state_dict."""
    ckpt = torch.load(pretrained_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # SimCLR ContrastiveModel stores backbone under "backbone." prefix
    backbone_sd = {}
    for k, v in ckpt.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v

    if not backbone_sd:
        backbone_sd = ckpt

    # Drop head weights — the classifier head is always freshly initialised
    backbone_sd = {k: v for k, v in backbone_sd.items()
                   if not k.startswith("linear.") and not k.startswith("fc.")}

    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    print(f"Pretrained backbone loaded from {pretrained_path}")
    if missing:
        print(f"  keys initialised from scratch: {missing}")
    if unexpected:
        print(f"  ignored checkpoint keys: {unexpected}")
    return model


def freeze_backbone(model):
    """Freeze every parameter except the final classification head."""
    for name, param in model.named_parameters():
        if not name.startswith("linear."):
            param.requires_grad = False
    model._freeze_backbone = True
    return model
