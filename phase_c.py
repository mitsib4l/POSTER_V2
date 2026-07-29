"""
Phase C - Knowledge Storage: Representational Evidence (Steps 9-13 of the FER
explainability pipeline).

Given the model architecture and the emotion-only labels available in this
project's datasets, this script implements Steps 9-13 with the following
scope and approximations, which should be reported alongside any results:

  Step 9  - CKA (exact, linear) between every pair of stages for each hook
            point below; SVCCA/PWCCA are approximated via PCA + CCA on top
            components ('svcca_approx'); parameter-update norms (||deltaW||)
            are computed per module group. Per-step gradient norms from
            training are not available post hoc.
  Step 10 - Linear probes (logistic/ridge regression, cross-validated) for
            emotion class at every hook point. Identity/dataset/pose/etc.
            probing only runs if a --metadata-csv is supplied; without one
            (the default for FER2013/RAF-DB ImageFolder splits) those probes
            are skipped and reported as such.
  Step 11 - Silhouette / Davies-Bouldin / Calinski-Harabasz / k-NN purity and
            PCA/t-SNE plots on the final pooled embedding, colored by class
            and by prediction correctness (no identity/dataset labels to
            color by, per the Step 10 limitation above).
  Step 12 - AU concepts are approximated by four 68-point-landmark geometric
            proxies (eye/mouth aspect ratio, brow raise, mouth-corner pull),
            not annotated Action Units. TCAV is a lightweight directional-
            derivative variant computed on pooled hidden states.
  Step 13 - Attention analysis is approximated via gradient x activation
            rollout across the two VIT blocks (native attention weights are
            not exposed as tensors by WindowAttentionGlobal/Attention), plus
            descriptive branch ablation (zeroing each pyramid branch's ffn
            output) to measure branch contribution.
"""

import os
import csv
import math
import random
import argparse
import time
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

from phase_a import build_model, load_model_from_checkpoint, unwrap_model, write_json
from phase_b import (
    build_eval_dataset,
    select_balanced_indices,
    collect_stage_checkpoints,
    denormalize_tensor,
    extract_facemesh_landmarks,
    normalize_map,
    get_logits,
    set_seed,
    ensure_dir,
    append_csv,
)

CONCEPT_NAMES = ["eye_aspect_ratio", "mouth_aspect_ratio", "brow_raise", "mouth_corner_pull"]

MODULE_GROUPS: Dict[str, List[str]] = {
    "face_landback": ["face_landback"],
    "ir_back": ["ir_back"],
    "fusion": ["attn1", "attn2", "attn3", "window1", "window2", "window3",
               "ffn1", "ffn2", "ffn3", "conv1", "conv2", "conv3"],
    "cross_embed": ["embed_q", "embed_k", "embed_v", "last_face_conv"],
    "vit": ["VIT"],
}


# ---------------------------------------------------------------------------
# Representation-hook pooling (Step 9)
# ---------------------------------------------------------------------------

def pool_channel_first(t: torch.Tensor) -> np.ndarray:
    return t.detach().mean(dim=(2, 3)).cpu().numpy()


def pool_channel_last(t: torch.Tensor) -> np.ndarray:
    return t.detach().mean(dim=(1, 2)).cpu().numpy()


def pool_tokens(t: torch.Tensor, mode: str) -> np.ndarray:
    t = t.detach()
    if mode == "cls":
        vec = t[:, 0]
    elif mode == "mean":
        vec = t.mean(dim=1)
    else:
        raise ValueError(f"Unknown token pooling mode: {mode}")
    return vec.cpu().numpy()


def pool_by_kind(t: torch.Tensor, kind: str) -> np.ndarray:
    if kind == "identity":
        return t.detach().cpu().numpy()
    if kind == "cf":
        return pool_channel_first(t)
    if kind == "cl":
        return pool_channel_last(t)
    if kind in ("cls", "mean"):
        return pool_tokens(t, kind)
    raise ValueError(f"Unknown pooling kind: {kind}")


def build_hook_module_map(base: nn.Module) -> Dict[str, Tuple[nn.Module, str]]:
    mapping: Dict[str, Tuple[nn.Module, str]] = {
        "fusion_stage1": (base.ffn1, "cl"),
        "fusion_stage2": (base.ffn2, "cl"),
        "fusion_stage3": (base.ffn3, "cl"),
        "embed_q": (base.embed_q, "cf"),
        "embed_k": (base.embed_k, "cf"),
        "embed_v": (base.embed_v, "mean"),
        "vit_norm_cls": (base.VIT.norm, "cls"),
        "vit_pooled_final": (base.VIT.se_block, "identity"),
    }
    for i, block in enumerate(base.VIT.blocks):
        mapping[f"vit_block{i}_attn"] = (block.attn, "mean")
        mapping[f"vit_block{i}_mlp"] = (block.mlp, "mean")
        mapping[f"vit_block{i}_cls"] = (block, "cls")
    return mapping


def register_representation_hooks(model: nn.Module) -> Tuple[Dict[str, np.ndarray], List]:
    base = unwrap_model(model)
    captured: Dict[str, np.ndarray] = {}
    handles = []

    def make_multi_hook(prefix):
        def hook(module, inputs, output):
            for i, tensor in enumerate(output, start=1):
                captured[f"{prefix}{i}"] = pool_channel_first(tensor)
        return hook

    handles.append(base.face_landback.register_forward_hook(make_multi_hook("landmark_stage")))
    handles.append(base.ir_back.register_forward_hook(make_multi_hook("ir_stage")))

    def make_hook(name, kind):
        def hook(module, inputs, output):
            captured[name] = pool_by_kind(output, kind)
        return hook

    for name, (module, kind) in build_hook_module_map(base).items():
        handles.append(module.register_forward_hook(make_hook(name, kind)))

    return captured, handles


def remove_hooks(handles: List) -> None:
    for handle in handles:
        handle.remove()


# ---------------------------------------------------------------------------
# Shared sample cache (landmarks + AU-proxy concepts are stage-independent)
# ---------------------------------------------------------------------------

# Point indices follow the standard 68-point ibug/dlib scheme (matches phase_b.py's
# LANDMARK_IDXS, produced by the face-alignment detector): 0-16 jaw, 17-21/22-26
# eyebrows, 36-41/42-47 eyes, 48-67 mouth (outer 48-59, inner 60-67).
def compute_au_proxy_concepts(pts: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if pts is None or pts.shape[0] < 68:
        return None

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    face_scale = dist(0, 16) + 1e-6

    # Standard eye-aspect-ratio (Soukupova & Cech): (|p2-p6| + |p3-p5|) / (2*|p1-p4|).
    left_ear = (dist(37, 41) + dist(38, 40)) / (2.0 * dist(36, 39) + 1e-6)
    right_ear = (dist(43, 47) + dist(44, 46)) / (2.0 * dist(42, 45) + 1e-6)
    eye_aspect_ratio = (left_ear + right_ear) / 2.0

    mouth_h = dist(51, 57)
    mouth_w = dist(48, 54) + 1e-6
    mouth_aspect_ratio = mouth_h / mouth_w

    brow_raise = ((dist(19, 37) + dist(24, 43)) / 2.0) / face_scale
    mouth_corner_pull = mouth_w / face_scale

    return {
        "eye_aspect_ratio": eye_aspect_ratio,
        "mouth_aspect_ratio": mouth_aspect_ratio,
        "brow_raise": brow_raise,
        "mouth_corner_pull": mouth_corner_pull,
    }


def build_sample_cache(dataset, indices: List[int]) -> List[Dict[str, object]]:
    cache = []
    for idx in indices:
        image_tensor, label = dataset[idx]
        sample_path = dataset.samples[idx][0]
        rgb_img = denormalize_tensor(image_tensor)
        pts = extract_facemesh_landmarks(rgb_img)
        concept_values = compute_au_proxy_concepts(pts)
        cache.append({
            "idx": int(idx),
            "image_tensor": image_tensor,
            "label": int(label),
            "sample_path": sample_path,
            "concept_values": concept_values,
        })
    return cache


def extract_stage_data(model: nn.Module, sample_cache: List[Dict[str, object]], device: torch.device) -> Dict[str, object]:
    captured, handles = register_representation_hooks(model)
    reps: Dict[str, List[np.ndarray]] = {}
    labels = []
    preds = []
    probs_list = []

    model.eval()
    with torch.no_grad():
        for sample in sample_cache:
            captured.clear()
            input_tensor = sample["image_tensor"].unsqueeze(0).to(device)
            logits = get_logits(model(input_tensor))
            prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

            for name, vec in captured.items():
                reps.setdefault(name, []).append(vec[0])

            labels.append(sample["label"])
            preds.append(int(np.argmax(prob)))
            probs_list.append(prob)

    remove_hooks(handles)

    reps_arrays = {name: np.stack(vectors, axis=0) for name, vectors in reps.items()}
    return {
        "representations": reps_arrays,
        "labels": np.array(labels, dtype=np.int64),
        "predictions": np.array(preds, dtype=np.int64),
        "probs": np.stack(probs_list, axis=0),
    }


# ---------------------------------------------------------------------------
# Step 9: representation similarity, shift magnitude, weight-update norms
# ---------------------------------------------------------------------------

def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.astype(np.float64).mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(y.T @ x, ord="fro") ** 2)
    denom = float(np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro"))
    return cross / denom if denom > 1e-12 else 0.0


def svcca_approx(x: np.ndarray, y: np.ndarray, max_components: int) -> float:
    n = x.shape[0]
    if n < 3:
        return 0.0
    k = max(1, min(max_components, n - 1, x.shape[1], y.shape[1]))
    x_c = x - x.mean(axis=0, keepdims=True)
    y_c = y - y.mean(axis=0, keepdims=True)
    try:
        x_pca = PCA(n_components=k, random_state=0).fit_transform(x_c)
        y_pca = PCA(n_components=k, random_state=0).fit_transform(y_c)
        cca = CCA(n_components=k, max_iter=1000)
        cca.fit(x_pca, y_pca)
        x_c2, y_c2 = cca.transform(x_pca, y_pca)
    except Exception:
        return 0.0
    correlations = []
    for i in range(k):
        a, b = x_c2[:, i], y_c2[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(correlations)) if correlations else 0.0


def representation_shift_magnitude(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.linalg.norm(x - y, axis=1)
    base = np.linalg.norm(x, axis=1) + 1e-12
    return float(np.mean(diff / base))


def mean_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(xn * yn, axis=1)))


def participation_ratio_dimensionality(X: np.ndarray) -> float:
    if X.shape[0] < 3:
        return float("nan")
    centered = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals ** 2))
    if s2 < 1e-12:
        return 0.0
    return (s1 ** 2) / s2


def run_representation_similarity(stage_data, stages, output_root: Path, args) -> None:
    layer_names = sorted(stage_data[stages[0]]["representations"].keys())

    sim_csv = output_root / "metrics" / "representation_similarity.csv"
    for layer in layer_names:
        for stage_a, stage_b in combinations(stages, 2):
            x = stage_data[stage_a]["representations"][layer]
            y = stage_data[stage_b]["representations"][layer]
            row = {
                "layer": layer,
                "stage_a": stage_a,
                "stage_b": stage_b,
                "cka": linear_cka(x, y),
                "svcca_approx": svcca_approx(x, y, args.svcca_components),
                "cosine": mean_cosine_similarity(x, y),
                "shift_magnitude": representation_shift_magnitude(x, y),
            }
            append_csv(sim_csv, row, fieldnames=list(row.keys()))

    stats_csv = output_root / "metrics" / "representation_stats.csv"
    for stage in stages:
        for layer in layer_names:
            X = stage_data[stage]["representations"][layer]
            row = {
                "stage": stage,
                "layer": layer,
                "n_samples": int(X.shape[0]),
                "mean_norm": float(np.mean(np.linalg.norm(X, axis=1))),
                "intrinsic_dimensionality": participation_ratio_dimensionality(X),
            }
            append_csv(stats_csv, row, fieldnames=list(row.keys()))


def load_state_dict_only(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def compute_weight_update_norms(
    state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for group_name, prefixes in MODULE_GROUPS.items():
        delta_sq = 0.0
        base_sq = 0.0
        count = 0
        for key, tensor_b in state_b.items():
            if not any(key.startswith(prefix) for prefix in prefixes):
                continue
            tensor_a = state_a.get(key)
            if tensor_a is None or tensor_a.shape != tensor_b.shape:
                continue
            diff = tensor_b.float() - tensor_a.float()
            delta_sq += float(torch.sum(diff * diff))
            base_sq += float(torch.sum(tensor_a.float() ** 2))
            count += tensor_b.numel()
        delta_norm = math.sqrt(delta_sq)
        base_norm = math.sqrt(base_sq) + 1e-12
        results[group_name] = {
            "delta_norm": delta_norm,
            "relative_delta_norm": delta_norm / base_norm,
            "num_params": count,
        }
    return results


def run_weight_update_norms(stage_ckpts, stages, device: torch.device, output_root: Path) -> None:
    state_cache = {stage: load_state_dict_only(str(stage_ckpts[stage]), device) for stage in stages}
    csv_path = output_root / "metrics" / "weight_update_norms.csv"
    for stage_a, stage_b in combinations(stages, 2):
        deltas = compute_weight_update_norms(state_cache[stage_a], state_cache[stage_b])
        for group_name, values in deltas.items():
            row = {"module_group": group_name, "stage_a": stage_a, "stage_b": stage_b, **values}
            append_csv(csv_path, row, fieldnames=list(row.keys()))


# ---------------------------------------------------------------------------
# Step 10: layer-wise linear probing (emotion + optional confounders)
# ---------------------------------------------------------------------------

def run_linear_probe_classification(X: np.ndarray, y: np.ndarray, folds: int) -> Optional[Dict[str, float]]:
    counts = np.bincount(y)
    valid_classes = counts[counts > 0]
    if len(valid_classes) < 2:
        return None
    eff_folds = min(folds, int(valid_classes.min()))
    if eff_folds < 2 or len(y) < eff_folds:
        return None
    skf = StratifiedKFold(n_splits=eff_folds, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    preds = cross_val_predict(clf, X, y, cv=skf)
    return {
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0) * 100.0),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds) * 100.0),
        "n_samples": int(len(y)),
        "n_folds": int(eff_folds),
    }


def run_linear_probe_regression(X: np.ndarray, y: np.ndarray, folds: int) -> Optional[Dict[str, float]]:
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    if len(y) < folds + 1:
        return None
    eff_folds = max(2, min(folds, len(y) // 2))
    if eff_folds < 2:
        return None
    kf = KFold(n_splits=eff_folds, shuffle=True, random_state=42)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    preds = cross_val_predict(reg, X, y, cv=kf)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"r2": float(r2), "n_samples": int(len(y)), "n_folds": int(eff_folds)}


def load_metadata_csv(path: str) -> Dict[str, object]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    if "sample_path" not in fieldnames:
        raise ValueError("--metadata-csv must contain a 'sample_path' column matching dataset image paths.")
    columns = [c for c in fieldnames if c != "sample_path"]
    lookup = {row["sample_path"]: row for row in rows}
    return {"lookup": lookup, "columns": columns}


def align_metadata_column(metadata_lookup: Dict[str, object], sample_paths: List[str], column: str):
    raw_values = []
    mask = np.zeros(len(sample_paths), dtype=bool)
    for i, path in enumerate(sample_paths):
        row = metadata_lookup["lookup"].get(path) or metadata_lookup["lookup"].get(str(Path(path).name))
        if row is None or row.get(column, "") == "":
            raw_values.append(None)
            continue
        raw_values.append(row[column])
        mask[i] = True

    present = [v for v in raw_values if v is not None]
    is_categorical = True
    if present:
        try:
            [float(v) for v in present]
            is_categorical = False
        except ValueError:
            is_categorical = True

    if is_categorical:
        encoder = LabelEncoder()
        encoded_present = encoder.fit_transform(present) if present else np.array([])
        values = np.zeros(len(sample_paths), dtype=np.int64)
        pointer = 0
        for i, v in enumerate(raw_values):
            if v is not None:
                values[i] = encoded_present[pointer]
                pointer += 1
    else:
        values = np.zeros(len(sample_paths), dtype=np.float64)
        pointer = 0
        for i, v in enumerate(raw_values):
            if v is not None:
                values[i] = float(present[pointer])
                pointer += 1
    return values, is_categorical, mask


def run_linear_probing(stage_data, stages, labels, metadata_lookup, sample_cache, output_root: Path, args) -> None:
    emotion_csv = output_root / "metrics" / "linear_probe_emotion.csv"
    for stage in stages:
        for layer, X in stage_data[stage]["representations"].items():
            result = run_linear_probe_classification(X, labels, args.probe_folds)
            if result is None:
                continue
            row = {"stage": stage, "layer": layer, "target": "emotion", **result}
            append_csv(emotion_csv, row, fieldnames=list(row.keys()))

    if metadata_lookup is None:
        return

    sample_paths = [s["sample_path"] for s in sample_cache]
    confounder_csv = output_root / "metrics" / "linear_probe_confounders.csv"
    for column in metadata_lookup["columns"]:
        values, is_categorical, mask = align_metadata_column(metadata_lookup, sample_paths, column)
        if int(mask.sum()) < args.probe_folds + 1:
            print(f"Skipping confounder probe for '{column}': not enough matched samples.")
            continue
        for stage in stages:
            for layer, X in stage_data[stage]["representations"].items():
                X_masked = X[mask]
                y_masked = values[mask]
                if is_categorical:
                    result = run_linear_probe_classification(X_masked, y_masked, args.probe_folds)
                else:
                    result = run_linear_probe_regression(X_masked, y_masked, args.probe_folds)
                if result is None:
                    continue
                row = {"stage": stage, "layer": layer, "target": column, **result}
                append_csv(confounder_csv, row, fieldnames=list(row.keys()))


# ---------------------------------------------------------------------------
# Step 11: embedding geometry
# ---------------------------------------------------------------------------

def compute_geometry_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    result: Dict[str, float] = {}
    unique_labels = np.unique(labels)
    if len(X) > len(unique_labels) >= 2:
        try:
            result["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            result["silhouette"] = float("nan")
        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            result["davies_bouldin"] = float("nan")
        try:
            result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except Exception:
            result["calinski_harabasz"] = float("nan")
    else:
        result["silhouette"] = float("nan")
        result["davies_bouldin"] = float("nan")
        result["calinski_harabasz"] = float("nan")

    k = min(5, len(X) - 1)
    if k >= 1:
        nn_model = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, indices = nn_model.kneighbors(X)
        purity = [
            float(np.mean(labels[neighbors[1:]] == labels[i]))
            for i, neighbors in enumerate(indices)
        ]
        result["knn_purity"] = float(np.mean(purity))
    else:
        result["knn_purity"] = float("nan")

    return result


def save_embedding_plot(X, labels, class_names, save_path: Path, title: str, correctness, method: str = "pca") -> None:
    if X.shape[0] < 3:
        return
    if method == "tsne":
        perplexity = max(2, min(30, X.shape[0] // 3))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    else:
        reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=140)
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=14)
    axes[0].set_title(f"{title} ({method.upper()}) by class")
    if len(class_names) <= 12:
        handles_legend, _ = scatter.legend_elements()
        unique_vals = sorted(np.unique(labels).tolist())
        axes[0].legend(handles_legend, [class_names[v] for v in unique_vals], fontsize=6, loc="best")

    axes[1].scatter(coords[:, 0], coords[:, 1], c=correctness, cmap="RdYlGn", s=14, vmin=0, vmax=1)
    axes[1].set_title(f"{title} ({method.upper()}) by correctness")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_geometry_analysis(stage_name: str, data: Dict[str, object], class_names: List[str], output_root: Path) -> None:
    labels = data["labels"]
    preds = data["predictions"]
    correctness = (labels == preds).astype(np.float64)

    csv_path = output_root / "metrics" / "embedding_geometry.csv"
    for layer, X in data["representations"].items():
        metrics = compute_geometry_metrics(X, labels)
        row = {"stage": stage_name, "layer": layer, "n_samples": int(X.shape[0]), **metrics}
        append_csv(csv_path, row, fieldnames=list(row.keys()))

    primary_layer = "vit_pooled_final"
    if primary_layer in data["representations"]:
        X = data["representations"][primary_layer]
        save_embedding_plot(
            X, labels, class_names,
            output_root / "plots" / f"{stage_name}_{primary_layer}_pca.png",
            title=f"{stage_name} {primary_layer}",
            correctness=correctness,
            method="pca",
        )
        if X.shape[0] >= 10:
            save_embedding_plot(
                X, labels, class_names,
                output_root / "plots" / f"{stage_name}_{primary_layer}_tsne.png",
                title=f"{stage_name} {primary_layer}",
                correctness=correctness,
                method="tsne",
            )


# ---------------------------------------------------------------------------
# Step 12: AU-proxy concept regression probing + lightweight TCAV
# ---------------------------------------------------------------------------

def run_concept_probing(stage_data, stages, sample_cache, output_root: Path, args) -> None:
    mask = np.array([s["concept_values"] is not None for s in sample_cache], dtype=bool)
    if mask.sum() < args.probe_folds + 1:
        print("Not enough landmark-detected samples for AU-proxy concept probing (Step 12); skipping.")
        return

    concept_targets = {
        name: np.array([
            (s["concept_values"][name] if s["concept_values"] is not None else np.nan)
            for s in sample_cache
        ])
        for name in CONCEPT_NAMES
    }

    csv_path = output_root / "metrics" / "concept_probe.csv"
    for stage in stages:
        for layer, X in stage_data[stage]["representations"].items():
            X_masked = X[mask]
            for concept in CONCEPT_NAMES:
                y_masked = concept_targets[concept][mask]
                result = run_linear_probe_regression(X_masked, y_masked, args.probe_folds)
                if result is None:
                    continue
                row = {"stage": stage, "layer": layer, "concept": concept, **result}
                append_csv(csv_path, row, fieldnames=list(row.keys()))


def run_tcav_analysis(model, stage_name, sample_cache, tcav_layers, device, args, output_root: Path) -> None:
    base = unwrap_model(model)
    hook_map = build_hook_module_map(base)

    eligible = [i for i, s in enumerate(sample_cache) if s["concept_values"] is not None]
    if len(eligible) < 10:
        print(f"[{stage_name}] Not enough landmark-detected samples for TCAV; skipping.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen = eligible[: min(args.tcav_samples, len(eligible))]

    csv_path = output_root / "metrics" / "concept_tcav.csv"

    for layer_name in tcav_layers:
        if layer_name not in hook_map:
            print(f"[{stage_name}] Skipping TCAV for unknown/ineligible layer '{layer_name}'.")
            continue
        module, pool_kind = hook_map[layer_name]

        pooled_grads = []
        pooled_reps = []
        used_indices = []
        for i in chosen:
            sample = sample_cache[i]
            captured = {}

            def hook(module_, inputs, output):
                output.retain_grad()
                captured["tensor"] = output

            handle = module.register_forward_hook(hook)
            model.zero_grad(set_to_none=True)
            input_tensor = sample["image_tensor"].unsqueeze(0).to(device)
            logits = get_logits(model(input_tensor))
            target_class = int(logits.argmax(dim=1).item())
            score = logits[:, target_class].sum()
            score.backward()
            handle.remove()

            tensor = captured["tensor"]
            grad = tensor.grad
            if grad is None:
                continue
            pooled_grads.append(pool_by_kind(grad, pool_kind)[0])
            pooled_reps.append(pool_by_kind(tensor, pool_kind)[0])
            used_indices.append(i)

        if len(pooled_grads) < 10:
            continue

        pooled_grads_arr = np.stack(pooled_grads, axis=0)
        pooled_reps_arr = np.stack(pooled_reps, axis=0)

        for concept in CONCEPT_NAMES:
            concept_values = np.array([sample_cache[i]["concept_values"][concept] for i in used_indices])
            median = float(np.median(concept_values))
            concept_labels = (concept_values > median).astype(np.int64)
            if len(np.unique(concept_labels)) < 2:
                continue
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            probe.fit(pooled_reps_arr, concept_labels)
            direction = probe.named_steps["logisticregression"].coef_[0]
            scale = probe.named_steps["standardscaler"].scale_
            direction = direction / (scale + 1e-12)
            direction = direction / (np.linalg.norm(direction) + 1e-12)

            sensitivities = pooled_grads_arr @ direction
            tcav_score = float(np.mean(sensitivities > 0))
            row = {
                "stage": stage_name,
                "layer": layer_name,
                "concept": concept,
                "tcav_score": tcav_score,
                "mean_sensitivity": float(np.mean(sensitivities)),
                "n_samples": int(len(used_indices)),
            }
            append_csv(csv_path, row, fieldnames=list(row.keys()))


# ---------------------------------------------------------------------------
# Step 13: gradient x activation rollout + branch ablation
# ---------------------------------------------------------------------------

def gradient_rollout(model, image_tensor: torch.Tensor, target_class: int, device: torch.device) -> Optional[np.ndarray]:
    base = unwrap_model(model)
    if not hasattr(base, "VIT") or not hasattr(base.VIT, "blocks"):
        return None

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            output.retain_grad()
            captured[idx] = output
        return hook

    for i, block in enumerate(base.VIT.blocks):
        handles.append(block.register_forward_hook(make_hook(i)))

    model.zero_grad(set_to_none=True)
    input_tensor = image_tensor.unsqueeze(0).to(device)
    logits = get_logits(model(input_tensor))
    score = logits[:, target_class].sum()
    score.backward()

    for handle in handles:
        handle.remove()

    relevance = None
    for i in sorted(captured.keys()):
        tensor = captured[i]
        grad = tensor.grad
        if grad is None:
            continue
        token_scores = (grad * tensor).detach().abs().sum(dim=-1)[0].cpu().numpy()
        token_scores = normalize_map(token_scores)
        relevance = token_scores if relevance is None else relevance * token_scores

    if relevance is None:
        return None
    return normalize_map(relevance)


def branch_relevance_breakdown(relevance: np.ndarray) -> Dict[str, float]:
    total = float(relevance.sum()) + 1e-12
    cls_mass = float(relevance[0] / total)
    patch = relevance[1:]
    thirds = np.array_split(patch, 3)
    patch_total = float(patch.sum()) + 1e-12
    return {
        "cls_mass": cls_mass,
        "branch_q_mass": float(thirds[0].sum() / patch_total),
        "branch_k_mass": float(thirds[1].sum() / patch_total),
        "branch_v_mass": float(thirds[2].sum() / patch_total),
    }


def save_relevance_plot(vector: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 3), dpi=140)
    plt.plot(vector, linewidth=1.2)
    plt.axvline(0, color="red", linestyle="--", linewidth=0.8, label="cls token")
    plt.title(title)
    plt.xlabel("Token index (0 = cls, 1-49 = q branch, 50-98 = k branch, 99-147 = v branch)")
    plt.ylabel("Relevance")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_attention_analysis(model, stage_name: str, sample_cache, device, args, output_root: Path) -> None:
    base = unwrap_model(model)
    rng = random.Random(args.seed)
    indices = list(range(len(sample_cache)))
    rng.shuffle(indices)
    chosen = indices[: min(args.rollout_samples, len(indices))]

    relevance_sum = None
    branch_masses = []
    for i in chosen:
        sample = sample_cache[i]
        with torch.no_grad():
            input_tensor = sample["image_tensor"].unsqueeze(0).to(device)
            target_class = int(get_logits(model(input_tensor)).argmax(dim=1).item())
        relevance = gradient_rollout(model, sample["image_tensor"], target_class, device)
        if relevance is None:
            continue
        relevance_sum = relevance if relevance_sum is None else relevance_sum + relevance
        branch_masses.append(branch_relevance_breakdown(relevance))

    if relevance_sum is not None and branch_masses:
        mean_relevance = relevance_sum / len(branch_masses)
        np.save(output_root / "metrics" / f"{stage_name}_token_rollout.npy", mean_relevance)
        save_relevance_plot(
            mean_relevance,
            output_root / "plots" / f"{stage_name}_token_rollout.png",
            title=f"{stage_name}: mean gradient x activation token rollout (VIT blocks)",
        )
        avg_branch = {key: float(np.mean([b[key] for b in branch_masses])) for key in branch_masses[0]}
        row = {"stage": stage_name, "n_samples": len(branch_masses), **avg_branch}
        append_csv(output_root / "metrics" / "token_rollout_summary.csv", row, fieldnames=list(row.keys()))

    baseline_preds = []
    baseline_targets = []
    for i in chosen:
        sample = sample_cache[i]
        with torch.no_grad():
            input_tensor = sample["image_tensor"].unsqueeze(0).to(device)
            logits = get_logits(model(input_tensor))
        baseline_preds.append(int(logits.argmax(dim=1).item()))
        baseline_targets.append(sample["label"])
    baseline_preds_arr = np.array(baseline_preds)
    baseline_targets_arr = np.array(baseline_targets)
    baseline_macro_f1 = float(
        f1_score(baseline_targets_arr, baseline_preds_arr, average="macro", zero_division=0) * 100.0
    )

    branch_modules = {"stage1_ffn": base.ffn1, "stage2_ffn": base.ffn2, "stage3_ffn": base.ffn3}
    ablation_csv = output_root / "metrics" / "branch_ablation.csv"
    for branch_name, module in branch_modules.items():
        def zero_hook(_module, _inputs, output):
            return torch.zeros_like(output)

        handle = module.register_forward_hook(zero_hook)
        ablated_preds = []
        for i in chosen:
            sample = sample_cache[i]
            with torch.no_grad():
                input_tensor = sample["image_tensor"].unsqueeze(0).to(device)
                logits = get_logits(model(input_tensor))
            ablated_preds.append(int(logits.argmax(dim=1).item()))
        handle.remove()

        ablated_preds_arr = np.array(ablated_preds)
        ablated_macro_f1 = float(
            f1_score(baseline_targets_arr, ablated_preds_arr, average="macro", zero_division=0) * 100.0
        )
        flip_rate = float(np.mean(ablated_preds_arr != baseline_preds_arr))
        row = {
            "stage": stage_name,
            "branch": branch_name,
            "baseline_macro_f1": round(baseline_macro_f1, 4),
            "ablated_macro_f1": round(ablated_macro_f1, 4),
            "macro_f1_drop": round(baseline_macro_f1 - ablated_macro_f1, 4),
            "prediction_flip_rate": round(flip_rate, 4),
            "n_samples": len(chosen),
        }
        append_csv(ablation_csv, row, fieldnames=list(row.keys()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--output-dir", type=str, default="./phase_c_outputs")
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--svcca-components", type=int, default=20)
    parser.add_argument("--tcav-samples", type=int, default=60)
    parser.add_argument("--tcav-layers", type=str, default="vit_pooled_final,vit_norm_cls")
    parser.add_argument("--rollout-samples", type=int, default=60)
    parser.add_argument("--metadata-csv", type=str, default=None,
                        help="Optional CSV with a 'sample_path' column plus confounder columns "
                             "(identity, dataset, pose, ...) for Step 10 selectivity probing.")
    parser.add_argument("--analyses", type=str,
                        default="representation,probing,geometry,concepts,attention",
                        help="Comma-separated subset of: representation,probing,geometry,concepts,attention")
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
    ensure_dir(output_root / "metrics")
    ensure_dir(output_root / "plots")

    dataset, class_names, num_classes = build_eval_dataset(args.data, args.split, args.data_type)
    print(f"Loaded {len(dataset)} samples with classes: {class_names}")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    analyses = {s.strip() for s in args.analyses.split(",") if s.strip()}
    tcav_layers = [s.strip() for s in args.tcav_layers.split(",") if s.strip()]
    stage_ckpts = collect_stage_checkpoints(args.checkpoints_root, stages, args.checkpoint_name)

    selected_indices = select_balanced_indices(
        dataset=dataset,
        samples_per_class=args.samples_per_class,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"Selected {len(selected_indices)} held-out samples shared across all stages (M0-M6 safeguard).")

    write_json(output_root / "logs" / "phase_c_config.json", vars(args))

    metadata_lookup = load_metadata_csv(args.metadata_csv) if args.metadata_csv else None
    if metadata_lookup is None:
        print("No --metadata-csv supplied: identity/dataset/pose confounder probing will be skipped "
              "(only emotion labels and landmark-derived AU-proxy concepts are available for this dataset).")

    print("Precomputing FaceMesh landmarks and AU-proxy concept targets shared across stages...")
    sample_cache = build_sample_cache(dataset, selected_indices)
    num_landmarked = sum(1 for s in sample_cache if s["concept_values"] is not None)
    print(f"Facial landmarks detected for {num_landmarked}/{len(sample_cache)} samples.")

    stage_data: Dict[str, Dict] = {}

    for stage_name in stages:
        print(f"\n=== Stage {stage_name}: loading checkpoint and extracting representations ===")
        model = build_model(num_classes=num_classes, device=device, use_dataparallel=False)
        model = load_model_from_checkpoint(model, str(stage_ckpts[stage_name]), device)
        model.eval()

        data = extract_stage_data(model, sample_cache, device)
        stage_data[stage_name] = data

        if "geometry" in analyses:
            run_geometry_analysis(stage_name, data, class_names, output_root)

        if "concepts" in analyses:
            run_tcav_analysis(model, stage_name, sample_cache, tcav_layers, device, args, output_root)

        if "attention" in analyses:
            run_attention_analysis(model, stage_name, sample_cache, device, args, output_root)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = stage_data[stages[0]]["labels"]

    if "representation" in analyses:
        print("\nComputing layer-wise representation similarity (CKA/SVCCA-approx/cosine/shift) across stages...")
        run_representation_similarity(stage_data, stages, output_root, args)
        run_weight_update_norms(stage_ckpts, stages, device, output_root)

    if "probing" in analyses:
        print("Running layer-wise linear probes for emotion (and confounders, if provided)...")
        run_linear_probing(stage_data, stages, labels, metadata_lookup, sample_cache, output_root, args)

    if "concepts" in analyses:
        print("Running AU-proxy concept regression probes...")
        run_concept_probing(stage_data, stages, sample_cache, output_root, args)

    print("\nPhase C complete.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
