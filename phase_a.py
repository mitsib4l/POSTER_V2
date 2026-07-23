import os
import json
import csv
import math
import time
import random
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt

from data_preprocessing.sam import SAM
from models.PosterV2_7cls import pyramid_trans_expr2

try:
    from torchsampler import ImbalancedDatasetSampler
except Exception:
    ImbalancedDatasetSampler = None

# For more repeatable splits and training
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

# Adapted from main.py
def load_checkpoint_state_dict(model: nn.Module, checkpoint_state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    model_state_dict = model.state_dict()

    if checkpoint_state_dict and all(key.startswith("module.") for key in checkpoint_state_dict.keys()):
        checkpoint_state_dict = {key[7:]: value for key, value in checkpoint_state_dict.items()}

    filtered_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict and model_state_dict[key].shape == value.shape:
            filtered_state_dict[key] = value

    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Missing keys ignored: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys ignored: {len(load_result.unexpected_keys)}")
    return model


def load_model_from_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return load_checkpoint_state_dict(model, state_dict)


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0

# extra metrics beyond accuracy and F1
def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(targets)

    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences >= start) & (confidences < end)

        if not np.any(mask):
            continue

        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes)[targets]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

# saves a visual confusion matrix
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 8), dpi=140)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=60, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

    thresh = cm_norm.max() * 0.5 if cm_norm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            if value > 0.01:
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# saves metric dictionaries
def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# writes one row per stage to a summary CSV
def append_csv(path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Data Transforms (adapted from the original main.py)
def build_transforms(data_type: str):
  # Fer2013
    if data_type == "fer2013":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.1)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
      # RAF-DB
        if data_type == "RAF-DB":
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(scale=(0.02, 0.1)),
            ])
        else:
          # CAER-S
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=1, scale=(0.05, 0.05)),
            ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return train_transform, test_transform

# STEP 1
# Loading folder structure and building Pytorch loaders (adapted from the original main.py)
def build_dataloaders(args):
    train_transform, test_transform = build_transforms(args.data_type)

    train_dir = Path(args.data) / args.train_split
    val_dir = Path(args.data) / args.val_split

    train_dataset = None
    if train_dir.is_dir():
        train_dataset = datasets.ImageFolder(str(train_dir), train_transform)

    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    val_dataset = datasets.ImageFolder(str(val_dir), test_transform)

    if train_dataset is not None:
        if len(train_dataset.classes) != len(val_dataset.classes):
            print("Warning: train/val class counts differ. The script will use the validation class count.")
        class_names = val_dataset.classes
        num_classes = len(val_dataset.classes)
    else:
        class_names = val_dataset.classes
        num_classes = len(val_dataset.classes)

    pin_memory = torch.cuda.is_available()

    if train_dataset is not None:
        if args.data_type == "AffectNet-7" and ImbalancedDatasetSampler is not None:
            train_loader = DataLoader(
                train_dataset,
                sampler=ImbalancedDatasetSampler(train_dataset),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=pin_memory,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=pin_memory,
            )
    else:
        train_loader = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    return train_dataset, val_dataset, train_loader, val_loader, class_names, num_classes

# Building the POSTER V2 network (pyramid_trans_expr2 comes from the original PosterV2_7cls.py)
def build_model(num_classes: int, device: torch.device, use_dataparallel: bool = True) -> nn.Module:
    model = pyramid_trans_expr2(img_size=224, num_classes=num_classes)
    model = model.to(device)
    if use_dataparallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def set_requires_grad_by_prefix(model: nn.Module, prefixes: List[str], train_all: bool = False) -> None:
    base = unwrap_model(model)

    if train_all:
        for param in base.parameters():
            param.requires_grad = True
        return

    for name, param in base.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)

# STEP 2
# Stage freezing logic
def apply_stage_freeze(model: nn.Module, stage_name: str) -> None:
    base = unwrap_model(model)

    if stage_name == "m0":
        prefixes = ["VIT.head"]
        set_requires_grad_by_prefix(model, prefixes, train_all=False)
    elif stage_name == "m1":
        prefixes = [
            "VIT.head",
            "VIT.se_block",
            "VIT.blocks",
            "VIT.norm",
        ]
        set_requires_grad_by_prefix(model, prefixes, train_all=False)
    elif stage_name == "m2":
        prefixes = [
            "VIT.head",
            "VIT.se_block",
            "VIT.blocks",
            "VIT.norm",
            "attn1",
            "attn2",
            "attn3",
            "ffn1",
            "ffn2",
            "ffn3",
            "window1",
            "window2",
            "window3",
            "conv1",
            "conv2",
            "conv3",
            "embed_q",
            "embed_k",
            "embed_v",
            "last_face_conv",
        ]
        set_requires_grad_by_prefix(model, prefixes, train_all=False)
    elif stage_name == "m3":
        prefixes = [
            "VIT.head",
            "VIT.se_block",
            "VIT.blocks",
            "VIT.norm",
            "attn1",
            "attn2",
            "attn3",
            "ffn1",
            "ffn2",
            "ffn3",
            "window1",
            "window2",
            "window3",
            "conv1",
            "conv2",
            "conv3",
            "embed_q",
            "embed_k",
            "embed_v",
            "last_face_conv",
            "ir_back",
            "face_landback",
        ]
        set_requires_grad_by_prefix(model, prefixes, train_all=False)
    elif stage_name == "m4":
        set_requires_grad_by_prefix(model, [], train_all=True)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"[{stage_name}] trainable parameters: {trainable:,} / {total:,}")

# Optimizer and scheduler for training setup (adapted from the original main.py)
def make_optimizer(model: nn.Module, args):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for this stage.")

    if args.optimizer == "adamw":
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == "adam":
        base_optimizer = torch.optim.Adam
    elif args.optimizer == "sgd":
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Unsupported optimizer")

    optimizer = SAM(
        trainable_params,
        base_optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rho=args.sam_rho,
        adaptive=False,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    return optimizer, scheduler

# adapted from the original main.py
def train_one_epoch(train_loader, model, criterion, optimizer, device, args, epoch_idx: int):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    for batch_idx, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        logits_second = model(images)
        loss_second = criterion(logits_second, target)
        loss_second.backward()
        optimizer.second_step(zero_grad=True)

        batch_size = images.size(0)
        running_loss += loss_second.item() * batch_size
        running_correct += (logits_second.argmax(dim=1) == target).sum().item()
        running_count += batch_size

        if batch_idx % args.print_freq == 0:
            print(
                f"Epoch {epoch_idx + 1:03d} | Batch {batch_idx:04d}/{len(train_loader):04d} | "
                f"loss {loss_second.item():.4f} | acc {100.0 * running_correct / max(1, running_count):.2f}"
            )

    avg_loss = running_loss / max(1, running_count)
    avg_acc = 100.0 * running_correct / max(1, running_count)
    return avg_loss, avg_acc

# STEP 3
# Evaluation adapted from the original main.py with new metrics added
@torch.no_grad()
def evaluate(val_loader, model, criterion, device, num_classes: int, class_names: List[str]):
    model.eval()

    all_probs = []
    all_targets = []
    all_preds = []
    total_loss = 0.0
    total_count = 0

    for images, target in val_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, target)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    avg_loss = total_loss / max(1, total_count)
    acc = accuracy_score(targets, preds) * 100.0
    bal_acc = balanced_accuracy_score(targets, preds) * 100.0
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0) * 100.0

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    report = classification_report(
        targets,
        preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    nll = float(np.mean(-np.log(np.clip(probs[np.arange(len(targets)), targets], 1e-12, 1.0))))
    brier = compute_brier(probs, targets, num_classes)
    ece = compute_ece(probs, targets)

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "confusion_matrix": cm,
        "classification_report": report,
        "probs": probs,
        "targets": targets,
        "preds": preds,
    }

# Checkpoint saving
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    stage_name: str,
    args,
) -> None:
    base = unwrap_model(model)
    checkpoint = {
        "epoch": epoch,
        "stage": stage_name,
        "state_dict": base.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "args": vars(args),
    }
    torch.save(checkpoint, path)

# Stage list and parsing stage epochs
def stage_configs():
    return [
        ("m0", "head_only"),
        ("m1", "upper_blocks"),
        ("m2", "fusion_blocks"),
        ("m3", "backbone_and_fusion"),
        ("m4", "full_finetune"),
    ]


def parse_stage_epochs(text: str, num_stages: int) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) == 1:
        values = values * num_stages
    if len(values) != num_stages:
        raise ValueError(f"--stage-epochs must have 1 value or {num_stages} comma-separated values.")
    return values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--data_type", type=str, default="RAF-DB",
                        choices=["RAF-DB", "AffectNet-7", "CAER-S", "fer2013"])
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="valid")
    parser.add_argument("--output-dir", type=str, default="./phase_a_outputs")
    parser.add_argument("--evaluate", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stage-epochs", type=str, default="10,10,10,10,10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--lr-gamma", type=float, default=0.98)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num-classes", type=int, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    run_tag = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    output_root = Path(args.output_dir) / run_tag
    ensure_dir(output_root)
    ensure_dir(output_root / "logs")
    ensure_dir(output_root / "checkpoints")
    ensure_dir(output_root / "metrics")
    ensure_dir(output_root / "plots")

    train_dataset, val_dataset, train_loader, val_loader, class_names, inferred_classes = build_dataloaders(args)
    num_classes = args.num_classes if args.num_classes is not None else inferred_classes
    if args.num_classes is not None and args.num_classes != inferred_classes:
        print(f"Warning: --num-classes={args.num_classes} differs from inferred class count {inferred_classes}.")
    if len(class_names) != num_classes:
        class_names = class_names[:num_classes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Classes: {class_names}")

    model = build_model(num_classes=num_classes, device=device, use_dataparallel=True)
    criterion = nn.CrossEntropyLoss()

    if args.evaluate is not None:
        model = load_model_from_checkpoint(model, args.evaluate, device)
        eval_result = evaluate(val_loader, model, criterion, device, num_classes, class_names)
        cm = eval_result.pop("confusion_matrix")
        report = eval_result.pop("classification_report")

        write_json(output_root / "metrics" / "evaluate_metrics.json", {
            **eval_result,
            "classification_report": report,
            "class_names": class_names,
        })
        plot_confusion_matrix(
            cm,
            class_names,
            output_root / "plots" / "evaluate_confusion_matrix.png",
            title="Evaluation confusion matrix",
        )
        np.savez_compressed(
            output_root / "metrics" / "evaluate_predictions.npz",
            probs=eval_result["probs"],
            targets=eval_result["targets"],
            preds=eval_result["preds"],
        )
        print(json.dumps({k: v for k, v in eval_result.items() if k not in ["probs", "targets", "preds"]}, indent=2))
        return

    if train_loader is None:
        raise RuntimeError("Training split not found. Make sure the dataset has a train folder.")

    stage_epoch_counts = parse_stage_epochs(args.stage_epochs, len(stage_configs()))
    summary_rows = []
    summary_csv = output_root / "metrics" / "phase_a_summary.csv"

    previous_stage_last = None
    model_start_checkpoint = None

    # TRAINING LOOP
    for stage_idx, ((stage_name, stage_label), stage_epochs) in enumerate(zip(stage_configs(), stage_epoch_counts)):
        stage_dir = output_root / stage_name
        ensure_dir(stage_dir)
        ensure_dir(stage_dir / "checkpoints")
        ensure_dir(stage_dir / "metrics")
        ensure_dir(stage_dir / "plots")

        print(f"\n=== Starting stage {stage_name} ({stage_label}) for {stage_epochs} epochs ===")

        if previous_stage_last is not None:
            model = load_model_from_checkpoint(model, str(previous_stage_last), device)

        apply_stage_freeze(model, stage_name)
        optimizer, scheduler = make_optimizer(model, args)

        best_acc = -1.0
        best_ckpt_path = stage_dir / "checkpoints" / "best.pth"
        last_ckpt_path = stage_dir / "checkpoints" / "last.pth"

        stage_history = []
        start_time = time.time()

        for epoch in range(stage_epochs):
            train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device, args, epoch)

            eval_result = evaluate(val_loader, model, criterion, device, num_classes, class_names)
            cm = eval_result.pop("confusion_matrix")
            report = eval_result.pop("classification_report")

            scheduler.step()

            epoch_metrics = {
                "stage": stage_name,
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(eval_result["loss"]),
                "val_acc": float(eval_result["accuracy"]),
                "val_balanced_acc": float(eval_result["balanced_accuracy"]),
                "val_macro_f1": float(eval_result["macro_f1"]),
                "val_weighted_f1": float(eval_result["weighted_f1"]),
                "val_nll": float(eval_result["nll"]),
                "val_brier": float(eval_result["brier"]),
                "val_ece": float(eval_result["ece"]),
            }
            stage_history.append(epoch_metrics)

            save_checkpoint(
                last_ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                best_metric=best_acc,
                stage_name=stage_name,
                args=args,
            )

            np.savez_compressed(
                stage_dir / "metrics" / f"epoch_{epoch + 1:03d}_predictions.npz",
                probs=eval_result["probs"],
                targets=eval_result["targets"],
                preds=eval_result["preds"],
            )

            write_json(
                stage_dir / "metrics" / f"epoch_{epoch + 1:03d}.json",
                {
                    **epoch_metrics,
                    "classification_report": report,
                    "class_names": class_names,
                },
            )

            plot_confusion_matrix(
                cm,
                class_names,
                stage_dir / "plots" / f"epoch_{epoch + 1:03d}_confusion_matrix.png",
                title=f"{stage_name} epoch {epoch + 1} confusion matrix",
            )

            current_val_acc = eval_result["accuracy"]
            if current_val_acc > best_acc:
                best_acc = current_val_acc
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch + 1,
                    best_metric=best_acc,
                    stage_name=stage_name,
                    args=args,
                )

            print(
                f"[{stage_name}] epoch {epoch + 1}/{stage_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
                f"val_acc={eval_result['accuracy']:.2f} val_f1={eval_result['macro_f1']:.2f} "
                f"val_ece={eval_result['ece']:.4f}"
            )

        elapsed = time.time() - start_time
        print(f"Stage {stage_name} completed in {elapsed / 60.0:.2f} minutes")

        with torch.no_grad():
            final_eval = evaluate(val_loader, model, criterion, device, num_classes, class_names)
        final_cm = final_eval.pop("confusion_matrix")
        final_report = final_eval.pop("classification_report")

        write_json(
            stage_dir / "metrics" / "stage_final.json",
            {
                "stage": stage_name,
                "stage_label": stage_label,
                "best_val_acc": float(best_acc),
                "final_metrics": {k: float(v) for k, v in final_eval.items() if isinstance(v, (int, float, np.floating))},
                "history": stage_history,
                "class_names": class_names,
                "classification_report": final_report,
            },
        )

        np.savez_compressed(
            stage_dir / "metrics" / "stage_final_predictions.npz",
            probs=final_eval["probs"],
            targets=final_eval["targets"],
            preds=final_eval["preds"],
        )

        plot_confusion_matrix(
            final_cm,
            class_names,
            stage_dir / "plots" / "stage_final_confusion_matrix.png",
            title=f"{stage_name} final confusion matrix",
        )

        summary_row = {
            "stage": stage_name,
            "label": stage_label,
            "epochs": stage_epochs,
            "best_val_acc": round(best_acc, 4),
            "final_val_acc": round(final_eval["accuracy"], 4),
            "final_macro_f1": round(final_eval["macro_f1"], 4),
            "final_balanced_acc": round(final_eval["balanced_accuracy"], 4),
            "final_nll": round(final_eval["nll"], 6),
            "final_brier": round(final_eval["brier"], 6),
            "final_ece": round(final_eval["ece"], 6),
            "last_checkpoint": str(last_ckpt_path),
            "best_checkpoint": str(best_ckpt_path),
        }
        summary_rows.append(summary_row)
        append_csv(summary_csv, summary_row, fieldnames=list(summary_row.keys()))

        previous_stage_last = last_ckpt_path

    write_json(output_root / "metrics" / "phase_a_summary.json", summary_rows)
    print("\nPhase A complete.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()