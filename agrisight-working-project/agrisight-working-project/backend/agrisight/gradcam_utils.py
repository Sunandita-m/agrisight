import numpy as np, torch
from PIL import Image
from . import data, models
from .config import IMG_SIZE

_tf = data.eval_tf()

def predict(model, img: Image.Image, device="cpu"):
    """Return (probabilities as numpy array, input tensor)."""
    x = _tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].cpu().numpy()
    return p, x

def gradcam_overlay(model, model_name, img: Image.Image, class_idx=None, device="cpu"):
    """Return (overlay RGB uint8 image, grayscale cam 0..1) for the predicted (or given) class."""
    # Import Grad-CAM lazily: predictions must remain available even when the
    # optional visualization package is not installed or cannot initialize.
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    p, x = predict(model, img, device)
    if class_idx is None: class_idx = int(p.argmax())
    x = x.clone().requires_grad_(True)
    with GradCAM(model=model, target_layers=models.target_layer(model, model_name)) as cam:
        g = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    rgb = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    return show_cam_on_image(rgb, g, use_rgb=True), g

def background_reliance(cam, border=0.2):
    """Share of Grad-CAM attention that falls on the outer border of the image
    (a rough proxy for 'model looks at the background, not the leaf')."""
    h, w = cam.shape; b = int(min(h, w) * border)
    mask = np.ones_like(cam, dtype=bool); mask[b:h - b, b:w - b] = False
    return float(cam[mask].sum() / (cam.sum() + 1e-8))
