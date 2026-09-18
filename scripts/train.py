"""
train.py
Generic training script used for BOTH the binary model and the stage model.

Usage:
    python scripts/train.py --task binary
    python scripts/train.py --task stage
    python scripts/train.py --task binary --tune          # run Optuna search first
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.dataset import build_dataloaders
from utils.stain_normalization import load_or_fit_reference
from models.hybrid_model import build_binary_model, build_stage_model


class FocalLoss(nn.Module):
    """Handles class imbalance (e.g. fewer stage3/adipose patches)."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none",
                                       label_smoothing=label_smoothing)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_task_config(task):
    if task == "binary":
        return config.BINARY_DIR, config.BINARY_CLASSES, build_binary_model
    elif task == "stage":
        return config.STAGE_DIR, config.STAGE_CLASSES, build_stage_model
    else:
        raise ValueError("task must be 'binary' or 'stage'")


def compute_class_weights(loader, num_classes):
    counts = np.zeros(num_classes)
    for _, labels in loader:
        for l in labels.numpy():
            counts[l] += 1
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_config(task, lr, weight_decay, dropout, batch_size,
                      num_epochs=None, trial=None, save_best=True):
    root_dir, class_names, build_fn = get_task_config(task)
    num_epochs = num_epochs or config.NUM_EPOCHS
    device = config.DEVICE

    stain_normalizer = None
    if os.path.exists(config.MACENKO_REFERENCE_PATCH):
        stain_normalizer = load_or_fit_reference(config.MACENKO_REFERENCE_PATCH)

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir, class_names, stain_normalizer, batch_size=batch_size
    )

    class_weights = compute_class_weights(train_loader, len(class_names)).to(device)
    model = build_fn(pretrained=True, dropout=dropout).to(device)

    criterion = FocalLoss(gamma=config.FOCAL_GAMMA, weight=class_weights,
                           label_smoothing=config.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_f1 = -1.0
    patience_counter = 0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{task}_best.pt")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # ---- validation ----
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"[{task}] epoch {epoch+1}/{num_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if trial is not None:
            trial.report(val_f1, epoch)
            import optuna
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            if save_best:
                torch.save({
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "config": {"lr": lr, "weight_decay": weight_decay,
                               "dropout": dropout, "batch_size": batch_size},
                }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"[{task}] early stopping at epoch {epoch+1}")
                break

    return best_val_f1


def run_optuna(task, n_trials=None):
    import optuna
    n_trials = n_trials or config.OPTUNA_TRIALS

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        val_f1 = train_one_config(task, lr, weight_decay, dropout, batch_size,
                                   num_epochs=15, trial=trial, save_best=False)
        return val_f1

    study = optuna.create_study(direction="maximize",
                                 pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    print(f"[{task}] Best trial: {study.best_trial.params}, F1={study.best_value:.4f}")
    return study.best_trial.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "stage"], required=True)
    parser.add_argument("--tune", action="store_true", help="run Optuna search first")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.tune:
        best_params = run_optuna(args.task)
        train_one_config(
            args.task,
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            dropout=best_params["dropout"],
            batch_size=best_params["batch_size"],
            num_epochs=args.epochs or config.NUM_EPOCHS,
        )
    else:
        train_one_config(
            args.task,
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            dropout=config.DROPOUT,
            batch_size=config.BATCH_SIZE,
            num_epochs=args.epochs or config.NUM_EPOCHS,
        )
