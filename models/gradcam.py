import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def generate_and_save_gradcam(model, input_tensor, orig_img, save_prefix, device='cuda'):
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
    logits = out if not isinstance(out, (tuple, list)) else out[0]
    pred_class = int(logits.argmax(dim=1).item())
    targets = [ClassifierOutputTarget(pred_class)]

    # Target layers
    backbone_layer = model.module.ir_back.body3[-1].res_layer[3] if hasattr(model, 'module') else model.ir_back.body3[-1].res_layer[3]
    fusion_layer = model.module.conv3 if hasattr(model, 'module') else model.conv3

    # Grad-CAM on backbone
    with GradCAM(model=model, target_layers=[backbone_layer], use_cuda=(device.startswith('cuda'))) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        vis = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
        Image.fromarray(vis).save(f'{save_prefix}_ir_back.png')

    # Grad-CAM on fusion conv3
    with GradCAM(model=model, target_layers=[fusion_layer], use_cuda=(device.startswith('cuda'))) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        vis = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
        Image.fromarray(vis).save(f'{save_prefix}_conv3.png')