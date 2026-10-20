import csv, io
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from .config import IMG_SIZE

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def read_csv(path, split=None):
    rows = []
    with open(path, encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            if split is None or r["split"] == split:
                rows.append((r["path"], int(r["label"])))
    return rows

def eval_tf():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

def train_tf(mode="basic"):
    """basic = standard augmentation. field = stronger augmentation that mimics real-field photos
    (lighting, blur, perspective, colour shifts, partial occlusion / background clutter)."""
    if mode == "basic":
        aug = [T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(15)]
        post = []
    elif mode == "field":
        aug = [T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
               T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2), T.RandomRotation(30),
               T.RandomPerspective(0.25, p=0.4),
               T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.08),
               T.RandomGrayscale(0.05),
               T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3)]
        post = [T.RandomErasing(p=0.4, scale=(0.02, 0.2), value="random")]
    else:
        raise ValueError(mode)
    return T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)] + post)

class LeafDS(Dataset):
    def __init__(self, rows, tf, perturb=None):
        self.rows, self.tf, self.perturb = rows, tf, perturb
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, y = self.rows[i]
        img = Image.open(p).convert("RGB")
        if self.perturb: img = apply_perturbation(img, self.perturb)
        return self.tf(img), y

# ---- robustness perturbations (applied to PIL images before the eval transform) ----
def apply_perturbation(img, name):
    from PIL import ImageEnhance
    if name == "bright_low":  return ImageEnhance.Brightness(img).enhance(0.5)
    if name == "bright_high": return ImageEnhance.Brightness(img).enhance(1.6)
    if name == "contrast_low": return ImageEnhance.Contrast(img).enhance(0.5)
    if name == "blur":        return img.filter(ImageFilter.GaussianBlur(3))
    if name == "noise":
        a = np.asarray(img).astype(np.float32)
        rng = np.random.default_rng(0)
        a = a + rng.normal(0, 25, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    raise ValueError(name)

PERTURBATIONS = ["bright_low", "bright_high", "contrast_low", "blur", "noise"]
