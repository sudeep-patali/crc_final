"""
evaluate.py
Loads a trained checkpoint and reports test-set metrics:
accuracy, macro-F1, per-class precision/recall, confusion matrix,
ROC-AUC (one-vs-rest for the 4-class stage model).
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import build_dataloaders
from utils.stain_normalization import load_or_fit_reference
from models.hybrid_model import build_binary_model, build_stage_model


def get_task_config(task):
    if task == "binary":
        return config.BINARY_DIR, config.BINARY_CLASSES, build_binary_model
    elif task == "stage":
        return config.STAGE_DIR, config.STAGE_CLASSES, build_stage_model
    raise ValueError("task must be 'binary' or 'stage'")


def evaluate(task):
    root_dir, class_names, build_fn = get_task_config(task)
    device = config.DEVICE

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{task}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_fn(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    stain_normalizer = None
    if os.path.exists(config.MACENKO_REFERENCE_PATCH):
        stain_normalizer = load_or_fit_reference(config.MACENKO_REFERENCE_PATCH)

    _, _, test_loader = build_dataloaders(root_dir, class_names, stain_normalizer)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print(f"\n===== {task.upper()} MODEL - TEST SET REPORT =====")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{task.capitalize()} Model Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(config.REPORT_DIR, f"{task}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved -> {cm_path}")

    # ROC-AUC (one-vs-rest)
    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])  # make it (N,2) for the loop below

    plt.figure(figsize=(6, 6))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{task.capitalize()} Model ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(config.REPORT_DIR, f"{task}_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    print(f"ROC curve saved -> {roc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "stage"], required=True)
    args = parser.parse_args()
    evaluate(args.task)
