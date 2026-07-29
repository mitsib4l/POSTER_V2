import os
import csv
import json
import time
import random
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp

from phase_a import build_transforms, build_model, load_model_from_checkpoint, unwrap_model

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None
    GradCAMPlusPlus = None
    ScoreCAM = None
    AblationCAM = None
    ClassifierOutputTarget = None
    show_cam_on_image = None


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FACE_MESH = mp.solutions.face_mesh

# Utility functions
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def append_csv(path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    x = x - float(x.min())
    max_val = float(x.max())
    if max_val > 1e-12:
        x = x / max_val
    return x.astype(np.float32)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    a = (a - a.mean()) / a_std
    b = (b - b.mean()) / b_std
    return float(np.mean(a * b))


def rank_simple(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return safe_pearson(rank_simple(a), rank_simple(b))


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def safe_l1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def entropy_from_probs(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def get_logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def denormalize_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    tensor = image_tensor.detach().cpu().float()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.numpy().transpose(1, 2, 0)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = array * STD + MEAN
    array = np.clip(array, 0.0, 1.0)
    return array.astype(np.float32)


def rgb_to_normalized_batch(rgb_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(rgb_batch.transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def save_heatmap_overlay(rgb_img: np.ndarray, heatmap: np.ndarray, save_path: Path) -> None:
    heatmap = normalize_map(heatmap)
    if show_cam_on_image is not None:
        vis = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        Image.fromarray(vis).save(save_path)
        return
    cmap = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = np.clip(0.55 * rgb_img + 0.45 * cmap, 0.0, 1.0)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(save_path)


def save_vector_plot(vector: np.ndarray, save_path: Path, title: str) -> None:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    plt.figure(figsize=(12, 3), dpi=140)
    plt.plot(vector, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Token index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
# STEP 4
# Checkpoint loading για μετατροπή σε explanations και comparisons
def build_eval_dataset(data_root: str, split: str, data_type: str):
    _, test_transform = build_transforms(data_type)
    eval_dir = Path(data_root) / split
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {eval_dir}")
    dataset = datasets.ImageFolder(str(eval_dir), test_transform)
    return dataset, dataset.classes, len(dataset.classes)

# Επιλογή δειγμάτων 
def select_balanced_indices(dataset, samples_per_class: int, max_samples: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(int(label), []).append(idx)

    selected = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_class])

    rng.shuffle(selected)
    return selected[:max_samples]

# Checkpoint loading (m0, m1, m2, m3, m4)
def collect_stage_checkpoints(checkpoints_root: str, stages: List[str], checkpoint_name: str) -> Dict[str, Path]:
    root = Path(checkpoints_root)
    stage_paths = {}
    for stage in stages:
        candidate = root / stage / "checkpoints" / checkpoint_name
        if not candidate.is_file():
            fallback = root / stage / "checkpoints" / "last.pth"
            if fallback.is_file():
                candidate = fallback
            else:
                raise FileNotFoundError(f"Missing checkpoint for stage {stage}: {candidate}")
        stage_paths[stage] = candidate
    return stage_paths

# Επιλογή layers προς ανάλυση (επεξήγηση)
def resolve_cam_layers(model) -> Dict[str, torch.nn.Module]:
    base = unwrap_model(model)
    if not hasattr(base, "ir_back") or not hasattr(base, "conv3"):
        raise AttributeError("Could not resolve target layers. Expected base.ir_back and base.conv3.")
    backbone_layer = base.ir_back.body3[-1].res_layer[3]
    fusion_layer = base.conv3
    return {"ir_back": backbone_layer, "conv3": fusion_layer}


def predict_step(model, input_tensor: torch.Tensor) -> Tuple[np.ndarray, int, float, float]:
    with torch.no_grad():
        logits = get_logits(model(input_tensor))
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    ent = entropy_from_probs(probs)
    return probs, pred, conf, ent

# STEP 6 
# Facial landmarks (χρησιμοποιήθηκε MediaPipe FaceMesh!!!)
LANDMARK_IDXS = {
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "nose": [1, 2, 4, 5, 6, 19, 94, 97, 98, 168, 195, 197, 327, 326],
    "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 291, 405, 321, 314, 17, 84, 181, 91, 146],
    "jaw": [234, 93, 132, 58, 172, 136, 150, 149, 176, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454],
}


def extract_facemesh_landmarks(rgb_img: np.ndarray) -> Optional[np.ndarray]:
    h, w = rgb_img.shape[:2]
    with FACE_MESH.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process((rgb_img * 255).astype(np.uint8))
        if not result.multi_face_landmarks:
            return None

        pts = []
        for lm in result.multi_face_landmarks[0].landmark:
            x = np.clip(lm.x * w, 0, w - 1)
            y = np.clip(lm.y * h, 0, h - 1)
            pts.append([x, y])
        return np.asarray(pts, dtype=np.float32)


def poly_mask(points: np.ndarray, shape) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if points is None or len(points) < 3:
        return mask.astype(np.float32)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(np.float32)


def box_mask(x1, y1, x2, y2, shape) -> np.ndarray:
    h, w = shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h, y2)))
    mask = np.zeros((h, w), dtype=np.float32)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def dilate_mask(mask, k=9):
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)


def union_masks(*masks):
    out = np.zeros_like(masks[0], dtype=np.float32)
    for m in masks:
        out = np.maximum(out, m.astype(np.float32))
    return out

# AU region mask construction από FaceMesh landmarks
def build_fallback_face_masks(shape) -> Dict[str, np.ndarray]:
    h, w = shape[:2]
    masks = {}

    upper = np.zeros((h, w), dtype=np.float32)
    upper[: int(0.33 * h), :] = 1.0
    masks["brows"] = upper

    eye = np.zeros((h, w), dtype=np.float32)
    eye[int(0.20 * h) : int(0.50 * h), :] = 1.0
    masks["eyes"] = eye

    nose = np.zeros((h, w), dtype=np.float32)
    nose[int(0.30 * h) : int(0.70 * h), int(0.33 * w) : int(0.67 * w)] = 1.0
    masks["nose"] = nose

    mouth = np.zeros((h, w), dtype=np.float32)
    mouth[int(0.55 * h) : int(0.85 * h), int(0.22 * w) : int(0.78 * w)] = 1.0
    masks["mouth"] = mouth

    cheek = np.zeros((h, w), dtype=np.float32)
    cheek[int(0.33 * h) : int(0.75 * h), :] = 1.0
    masks["cheeks"] = cheek

    chin = np.zeros((h, w), dtype=np.float32)
    chin[int(0.70 * h) :, int(0.25 * w) : int(0.75 * w)] = 1.0
    masks["chin"] = chin

    jaw = np.zeros((h, w), dtype=np.float32)
    jaw[int(0.62 * h) :, :] = 1.0
    masks["jaw"] = jaw

    masks["AU1_AU2"] = upper
    masks["AU4"] = upper
    masks["AU6"] = cheek
    masks["AU9"] = nose
    masks["AU12"] = mouth
    masks["AU15"] = mouth
    masks["AU17"] = chin
    masks["full_face"] = np.ones((h, w), dtype=np.float32)

    return masks
# πρόσωπο --> landmarks --> regions --> attribution overlap

def build_au_region_masks(pts: np.ndarray, shape) -> Dict[str, np.ndarray]:
    left_brow = poly_mask(pts[LANDMARK_IDXS["left_eyebrow"]], shape)
    right_brow = poly_mask(pts[LANDMARK_IDXS["right_eyebrow"]], shape)
    left_eye = poly_mask(pts[LANDMARK_IDXS["left_eye"]], shape)
    right_eye = poly_mask(pts[LANDMARK_IDXS["right_eye"]], shape)
    nose = poly_mask(pts[LANDMARK_IDXS["nose"]], shape)
    mouth = poly_mask(pts[LANDMARK_IDXS["mouth"]], shape)
    jaw = poly_mask(pts[LANDMARK_IDXS["jaw"]], shape)

    brow_all = union_masks(left_brow, right_brow)
    eye_all = union_masks(left_eye, right_eye)

    brow_pts = np.vstack([pts[LANDMARK_IDXS["left_eyebrow"]], pts[LANDMARK_IDXS["right_eyebrow"]]])
    mouth_pts = np.vstack([pts[LANDMARK_IDXS["mouth"]]])

    brow_y_min = float(np.min(brow_pts[:, 1]))
    brow_y_max = float(np.max(brow_pts[:, 1]))
    mouth_y_min = float(np.min(mouth_pts[:, 1]))
    mouth_y_max = float(np.max(mouth_pts[:, 1]))

    h, w = shape[:2]

    brow_band = box_mask(
        x1=0,
        y1=brow_y_min - 0.10 * h,
        x2=w,
        y2=brow_y_max + 0.05 * h,
        shape=shape,
    )

    upper_eye_band = box_mask(
        x1=0,
        y1=max(0, brow_y_max - 0.02 * h),
        x2=w,
        y2=max(1, mouth_y_min * 0.55),
        shape=shape,
    )

    left_eye_outer = pts[33]
    right_eye_outer = pts[263]
    left_mouth_corner = pts[61]
    right_mouth_corner = pts[291] if pts.shape[0] > 291 else pts[270]

    left_cheek = box_mask(
        x1=min(left_eye_outer[0], left_mouth_corner[0]) - 0.08 * w,
        y1=left_eye_outer[1],
        x2=left_mouth_corner[0] + 0.06 * w,
        y2=mouth_y_min + 0.10 * h,
        shape=shape,
    )

    right_cheek = box_mask(
        x1=right_mouth_corner[0] - 0.06 * w,
        y1=right_eye_outer[1],
        x2=max(right_eye_outer[0], right_mouth_corner[0]) + 0.08 * w,
        y2=mouth_y_min + 0.10 * h,
        shape=shape,
    )

    nose_band = box_mask(
        x1=float(np.min(pts[LANDMARK_IDXS["nose"]][:, 0])) - 0.05 * w,
        y1=float(np.min(pts[LANDMARK_IDXS["nose"]][:, 1])) - 0.03 * h,
        x2=float(np.max(pts[LANDMARK_IDXS["nose"]][:, 0])) + 0.05 * w,
        y2=float(np.max(pts[LANDMARK_IDXS["nose"]][:, 1])) + 0.05 * h,
        shape=shape,
    )

    mouth_corner_left = box_mask(
        x1=left_mouth_corner[0] - 0.08 * w,
        y1=mouth_y_min - 0.05 * h,
        x2=left_mouth_corner[0] + 0.08 * w,
        y2=mouth_y_max + 0.08 * h,
        shape=shape,
    )
    mouth_corner_right = box_mask(
        x1=right_mouth_corner[0] - 0.08 * w,
        y1=mouth_y_min - 0.05 * h,
        x2=right_mouth_corner[0] + 0.08 * w,
        y2=mouth_y_max + 0.08 * h,
        shape=shape,
    )
    mouth_corner_mask = union_masks(mouth_corner_left, mouth_corner_right)

    chin_band = box_mask(
        x1=0.25 * w,
        y1=mouth_y_max + 0.04 * h,
        x2=0.75 * w,
        y2=0.98 * h,
        shape=shape,
    )

    masks = {
        "brows": dilate_mask(brow_all, 9),
        "eyes": dilate_mask(eye_all, 7),
        "nose": dilate_mask(nose_band, 7),
        "mouth": dilate_mask(mouth, 7),
        "cheeks": dilate_mask(union_masks(left_cheek, right_cheek), 9),
        "chin": dilate_mask(chin_band, 7),
        "jaw": dilate_mask(jaw, 7),

        "AU1_AU2": dilate_mask(union_masks(brow_all, brow_band), 11),
        "AU4": dilate_mask(union_masks(brow_all, upper_eye_band), 9),
        "AU6": dilate_mask(union_masks(left_cheek, right_cheek), 9),
        "AU9": dilate_mask(nose_band, 9),
        "AU12": dilate_mask(mouth_corner_mask, 9),
        "AU15": dilate_mask(mouth_corner_mask, 9),
        "AU17": dilate_mask(chin_band, 9),

        "full_face": dilate_mask(union_masks(brow_all, eye_all, nose, mouth, chin_band, jaw), 5),
    }
    return masks

# Για overlap (υπολογίζει πόση saliency πέφτει μέσα σε κάθε region mask)
def region_mass(saliency_map: np.ndarray, mask: np.ndarray) -> float:
    sal = np.asarray(saliency_map, dtype=np.float32)
    sal = sal - float(sal.min())
    if float(sal.max()) > 1e-12:
        sal = sal / float(sal.max())
    mask = mask.astype(np.float32)
    denom = float(sal.sum()) + 1e-12
    return float((sal * mask).sum() / denom)


def compare_maps(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size:
        min_len = min(a.size, b.size)
        a = a[:min_len]
        b = b[:min_len]
    return {
        "pearson": safe_pearson(a, b),
        "spearman": safe_spearman(a, b),
        "cosine": safe_cosine(a, b),
        "l1": safe_l1(a, b),
        "mse": float(np.mean((a - b) ** 2)),
    }


def compare_vectors(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size:
        min_len = min(a.size, b.size)
        a = a[:min_len]
        b = b[:min_len]
    return {
        "pearson": safe_pearson(a, b),
        "spearman": safe_spearman(a, b),
        "cosine": safe_cosine(a, b),
        "l1": safe_l1(a, b),
    }

# STEP 4
# Explainers
def run_cam_method(model, input_tensor, target_class: int, cam_layer, cam_type: str) -> np.ndarray:
    if GradCAM is None:
        raise RuntimeError("pytorch-grad-cam is not installed. Install it to use CAM methods.")
    cam_registry = {
        "gradcam": GradCAM,
        "gradcampp": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
    }
    cam_cls = cam_registry.get(cam_type.lower())
    if cam_cls is None:
        raise ValueError(f"Unsupported CAM type: {cam_type}")
    targets = [ClassifierOutputTarget(target_class)]
    with cam_cls(model=model, target_layers=[cam_layer]) as cam:
        cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
    return normalize_map(cam_map)


def integrated_gradients(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    steps: int = 32,
    baseline: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model.eval()
    x = input_tensor.detach()
    base = torch.zeros_like(x) if baseline is None else baseline.detach().clone()
    total_grad = torch.zeros_like(x)

    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)[1:]
    for alpha in alphas:
        scaled = base + alpha * (x - base)
        scaled.requires_grad_(True)
        logits = get_logits(model(scaled))
        score = logits[:, target_class].sum()
        grad = torch.autograd.grad(score, scaled, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / float(steps)
    attr = (x - base) * avg_grad
    heatmap = attr.abs().sum(dim=1)[0].detach().cpu().numpy()
    return normalize_map(heatmap)


def smoothgrad(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    samples: int = 20,
    noise_std: float = 0.15,
) -> np.ndarray:
    model.eval()
    x = input_tensor.detach()
    total = torch.zeros_like(x)

    for _ in range(samples):
        noisy = x + torch.randn_like(x) * noise_std
        noisy.requires_grad_(True)
        logits = get_logits(model(noisy))
        score = logits[:, target_class].sum()
        grad = torch.autograd.grad(score, noisy, retain_graph=False, create_graph=False)[0]
        total += grad.abs().detach()

    heatmap = total.mean(dim=1)[0].detach().cpu().numpy()
    return normalize_map(heatmap)

# STEP 7 (να ποροσθέσω blur, inpainting, random controls, ROAD/ROAR, face-part cropping, landmark jitter)
def occlusion_sensitivity(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    patch_size: int = 32,
    stride: int = 16,
    baseline_value: float = 0.0,
) -> np.ndarray:
    model.eval()
    x = input_tensor.detach()
    h = x.shape[2]
    w = x.shape[3]
    with torch.no_grad():
        base_logits = get_logits(model(x))
        base_score = float(base_logits[0, target_class].item())

    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for xx in range(0, w, stride):
            occluded = x.clone()
            y2 = min(h, y + patch_size)
            x2 = min(w, xx + patch_size)
            occluded[:, :, y:y2, xx:x2] = baseline_value
            with torch.no_grad():
                logits = get_logits(model(occluded))
                score = float(logits[0, target_class].item())
            drop = base_score - score
            heatmap[y:y2, xx:x2] += drop
            counts[y:y2, xx:x2] += 1.0

    heatmap = np.divide(heatmap, np.maximum(counts, 1.0), out=np.zeros_like(heatmap), where=counts > 0)
    return normalize_map(heatmap)


def token_attribution(model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    base = unwrap_model(model)
    if not hasattr(base, "VIT") or not hasattr(base.VIT, "blocks"):
        raise AttributeError("Token attribution requires base.VIT.blocks to exist.")

    captured = {}

    def hook(_, __, output):
        captured["tokens"] = output
        output.retain_grad()

    handle = base.VIT.blocks.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = get_logits(model(input_tensor))
        score = logits[:, target_class].sum()
        score.backward()
        tokens = captured.get("tokens", None)
        grads = None if tokens is None else tokens.grad
        if tokens is None or grads is None:
            raise RuntimeError("Failed to capture token activations/gradients.")
        scores = grads.abs().mean(dim=-1)[0].detach().cpu().numpy()
        if scores.shape[0] > 1:
            scores = scores[1:]
        return normalize_map(scores)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def save_vector_plot(vector: np.ndarray, save_path: Path) -> None:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    plt.figure(figsize=(12, 3), dpi=140)
    plt.plot(vector, linewidth=1.2)
    plt.title("Transformer token attribution")
    plt.xlabel("Token index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# STEP 5 AND STEP 8
# Central Execution loop
def process_sample(
    sample_idx: int,
    sample_path: str,
    image_tensor: torch.Tensor,
    label: int,
    class_name: str,
    rgb_img: np.ndarray,
    stage_models: Dict[str, torch.nn.Module],
    stage_layers: Dict[str, Dict[str, torch.nn.Module]],
    methods: List[str],
    device: torch.device,
    args,
    sample_root: Path,
) -> Tuple[Dict[str, Dict], List[Dict[str, object]]]:
    sample_record = {
        "sample_idx": int(sample_idx),
        "sample_path": sample_path,
        "true_label": int(label),
        "class_name": class_name,
        "stages": {},
    }

    comparison_rows = []
    input_tensor = image_tensor.unsqueeze(0).to(device)

    pts = extract_facemesh_landmarks(rgb_img)
    if pts is None:
        region_masks = build_fallback_face_masks(rgb_img.shape)
        landmarks_found = False
    else:
        region_masks = build_au_region_masks(pts, rgb_img.shape)
        landmarks_found = True

    stage_store = {}

    for stage_name, model in stage_models.items():
        stage_dir = sample_root / stage_name
        ensure_dir(stage_dir)

        probs, pred, conf, ent = predict_step(model, input_tensor)
        target_class = int(label) if args.target_mode == "true" else int(pred)
        stage_info = {
            "prediction": pred,
            "confidence": conf,
            "entropy": ent,
            "target_class": target_class,
            "probabilities": probs.tolist(),
            "landmarks_found": landmarks_found,
            "methods": {},
        }

        for method in methods:
            method_dir = stage_dir / method
            ensure_dir(method_dir)

            if method == "gradcam_ir_back":
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["ir_back"], "gradcam")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "gradcam_conv3":
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["conv3"], "gradcam")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "gradcampp_ir_back":
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["ir_back"], "gradcampp")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "gradcampp_conv3":
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["conv3"], "gradcampp")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "scorecam_ir_back":
                if ScoreCAM is None:
                    raise RuntimeError("ScoreCAM requested, but pytorch-grad-cam is not installed.")
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["ir_back"], "scorecam")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "scorecam_conv3":
                if ScoreCAM is None:
                    raise RuntimeError("ScoreCAM requested, but pytorch-grad-cam is not installed.")
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["conv3"], "scorecam")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "ablationcam_ir_back":
                if AblationCAM is None:
                    raise RuntimeError("AblationCAM requested, but pytorch-grad-cam is not installed.")
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["ir_back"], "ablationcam")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "ablationcam_conv3":
                if AblationCAM is None:
                    raise RuntimeError("AblationCAM requested, but pytorch-grad-cam is not installed.")
                heatmap = run_cam_method(model, input_tensor, target_class, stage_layers[stage_name]["conv3"], "ablationcam")
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "ig":
                heatmap = integrated_gradients(
                    model=model,
                    input_tensor=input_tensor,
                    target_class=target_class,
                    steps=args.ig_steps,
                )
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "smoothgrad":
                heatmap = smoothgrad(
                    model=model,
                    input_tensor=input_tensor,
                    target_class=target_class,
                    samples=args.smoothgrad_samples,
                    noise_std=args.smoothgrad_noise,
                )
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "occlusion":
                heatmap = occlusion_sensitivity(
                    model=model,
                    input_tensor=input_tensor,
                    target_class=target_class,
                    patch_size=args.occlusion_patch,
                    stride=args.occlusion_stride,
                    baseline_value=0.0,
                )
                np.save(method_dir / "heatmap.npy", heatmap)
                save_heatmap_overlay(rgb_img, heatmap, method_dir / "overlay.png")
                stage_info["methods"][method] = {
                    "kind": "map",
                    "heatmap_path": str(method_dir / "heatmap.npy"),
                    "overlay_path": str(method_dir / "overlay.png"),
                    "region_mass": {k: region_mass(heatmap, v) for k, v in region_masks.items()},
                }

            elif method == "tokenattr":
                vector = token_attribution(model=model, input_tensor=input_tensor, target_class=target_class)
                np.save(method_dir / "vector.npy", vector)
                save_vector_plot(vector, method_dir / "vector.png")
                stage_info["methods"][method] = {
                    "kind": "vector",
                    "vector_path": str(method_dir / "vector.npy"),
                    "plot_path": str(method_dir / "vector.png"),
                }

        stage_store[stage_name] = stage_info

        with (stage_dir / "stage_result.json").open("w", encoding="utf-8") as f:
            json.dump(stage_info, f, indent=2, default=json_default)

    stage_names = list(stage_store.keys())

    for method in methods:
        if method == "tokenattr":
            for s1, s2 in combinations(stage_names, 2):
                v1 = np.load(stage_store[s1]["methods"][method]["vector_path"])
                v2 = np.load(stage_store[s2]["methods"][method]["vector_path"])
                comp = compare_vectors(v1, v2)
                comparison_rows.append(
                    {
                        "sample_idx": int(sample_idx),
                        "sample_path": sample_path,
                        "method": method,
                        "stage_a": s1,
                        "stage_b": s2,
                        **comp,
                    }
                )
        else:
            for s1, s2 in combinations(stage_names, 2):
                if method not in stage_store[s1]["methods"] or method not in stage_store[s2]["methods"]:
                    continue
                m1 = np.load(stage_store[s1]["methods"][method]["heatmap_path"])
                m2 = np.load(stage_store[s2]["methods"][method]["heatmap_path"])
                comp = compare_maps(m1, m2)
                comparison_rows.append(
                    {
                        "sample_idx": int(sample_idx),
                        "sample_path": sample_path,
                        "method": method,
                        "stage_a": s1,
                        "stage_b": s2,
                        **comp,
                    }
                )

    sample_record["stages"] = stage_store
    return sample_record, comparison_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--data_type", type=str, default="fer2013",
                        choices=["RAF-DB", "AffectNet-7", "CAER-S", "fer2013"])
    parser.add_argument("--checkpoints-root", type=str, required=True,
                        help="Root folder produced by phase A, containing m0..m4 subfolders")
    parser.add_argument("--checkpoint-name", type=str, default="best.pth")
    parser.add_argument("--stages", type=str, default="m0,m1,m2,m3,m4")
    parser.add_argument("--output-dir", type=str, default="./phase_b_outputs")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--methods", type=str,
                        default="gradcam_ir_back,gradcam_conv3,gradcampp_ir_back,gradcampp_conv3,ig,smoothgrad,occlusion,tokenattr")
    parser.add_argument("--target-mode", type=str, default="true", choices=["true", "pred"])
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--smoothgrad-samples", type=int, default=20)
    parser.add_argument("--smoothgrad-noise", type=float, default=0.15)
    parser.add_argument("--occlusion-patch", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_tag = time.strftime("%m-%d-%H-%M-%S")
    output_root = Path(args.output_dir) / run_tag
    ensure_dir(output_root)
    ensure_dir(output_root / "logs")
    ensure_dir(output_root / "results")
    ensure_dir(output_root / "comparisons")

    dataset, class_names, num_classes = build_eval_dataset(args.data, args.split, args.data_type)
    print(f"Loaded {len(dataset)} validation samples with classes: {class_names}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    stage_ckpts = collect_stage_checkpoints(args.checkpoints_root, stages, args.checkpoint_name)

    selected_indices = select_balanced_indices(
        dataset=dataset,
        samples_per_class=args.samples_per_class,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"Selected {len(selected_indices)} samples for Phase B")

    stage_models: Dict[str, torch.nn.Module] = {}
    stage_layers: Dict[str, Dict[str, torch.nn.Module]] = {}

    for stage_name in stages:
        print(f"Loading stage {stage_name} from {stage_ckpts[stage_name]}")
        model = build_model(num_classes=num_classes, device=device, use_dataparallel=False)
        model = load_model_from_checkpoint(model, str(stage_ckpts[stage_name]), device)
        model.eval()
        stage_models[stage_name] = model
        stage_layers[stage_name] = resolve_cam_layers(model)

    all_samples = []
    all_comparisons = []

    for idx in selected_indices:
        image_tensor, label = dataset[idx]
        sample_path = dataset.samples[idx][0]
        sample_name = f"{idx:05d}_{Path(sample_path).stem}"
        sample_root = output_root / "results" / sample_name
        ensure_dir(sample_root)

        rgb_img = denormalize_tensor(image_tensor)
        sample_record, comparison_rows = process_sample(
            sample_idx=idx,
            sample_path=sample_path,
            image_tensor=image_tensor,
            label=int(label),
            class_name=class_names[int(label)],
            rgb_img=rgb_img,
            stage_models=stage_models,
            stage_layers=stage_layers,
            methods=methods,
            device=device,
            args=args,
            sample_root=sample_root,
        )
        all_samples.append(sample_record)
        all_comparisons.extend(comparison_rows)

        with (sample_root / "sample_result.json").open("w", encoding="utf-8") as f:
            json.dump(sample_record, f, indent=2, default=json_default)

        print(f"Processed sample {sample_name}")

    with (output_root / "logs" / "phase_b_samples.json").open("w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, default=json_default)

    comparisons_csv = output_root / "comparisons" / "stage_comparisons.csv"
    if all_comparisons:
        fieldnames = list(all_comparisons[0].keys())
        for row in all_comparisons:
            append_csv(comparisons_csv, row, fieldnames=fieldnames)

    summary_rows = []
    for sample in all_samples:
        base_row = {
            "sample_idx": sample["sample_idx"],
            "sample_path": sample["sample_path"],
            "true_label": sample["true_label"],
            "class_name": sample["class_name"],
        }
        for stage_name, stage_info in sample["stages"].items():
            row = {
                **base_row,
                "stage": stage_name,
                "prediction": stage_info["prediction"],
                "confidence": stage_info["confidence"],
                "entropy": stage_info["entropy"],
                "target_class": stage_info["target_class"],
                "landmarks_found": stage_info.get("landmarks_found", False),
            }
            summary_rows.append(row)

    summary_csv = output_root / "logs" / "phase_b_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        for row in summary_rows:
            append_csv(summary_csv, row, fieldnames=fieldnames)

    print("\nPhase B complete.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()