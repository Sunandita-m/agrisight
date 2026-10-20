import torch, torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):
    """Simple 4-block CNN trained from scratch (the 'baseline')."""
    def __init__(self, n_classes):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))

NAMES = ["baseline", "mobilenet_v3", "efficientnet_b0", "resnet50"]

def build(name, n_classes, pretrained=True):
    if name == "baseline":
        return BaselineCNN(n_classes)
    if name == "mobilenet_v3":
        m = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "resnet50":
        m = models.resnet50(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(name)
    return m

def head_params(m, name):
    if name == "resnet50": return list(m.fc.parameters())
    if name == "baseline": return list(m.parameters())
    return list(m.classifier.parameters())

def last_block_params(m, name):
    """Parameters unfrozen in stage 2 (partial fine-tuning)."""
    if name == "resnet50": return list(m.layer4.parameters())
    if name == "mobilenet_v3": return [p for b in m.features[-4:] for p in b.parameters()]
    if name == "efficientnet_b0": return [p for b in m.features[-3:] for p in b.parameters()]
    return list(m.parameters())

def target_layer(m, name):
    """Layer used for Grad-CAM."""
    if name == "resnet50": return [m.layer4[-1]]
    if name == "baseline": return [m.features[-1][0]]
    return [m.features[-1]]

def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck["model"], len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m, ck
