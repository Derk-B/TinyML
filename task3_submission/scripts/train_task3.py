#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlp_hackathon.baseline_model import BaselineMLP


def euclidean_errors_cm(pred, true):
    return np.linalg.norm(pred - true, axis=1)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    train_csv = ROOT / "data" / "train_clean_6x6_8cm.csv"
    val_csv = ROOT / "data" / "validation_clean_6x6_8cm.csv"
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    feature_columns = [f"led_{i}" for i in range(36)]

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    y_train_cm = train_df[["x", "y"]].to_numpy(dtype=np.float32) / 10.0

    X_val_raw = val_df[feature_columns].to_numpy(dtype=np.float32)
    y_val_cm = val_df[["x", "y"]].to_numpy(dtype=np.float32) / 10.0

    eps = 1e-6

    X_train_log = np.log1p(np.maximum(X_train_raw, 0.0))
    X_val_log = np.log1p(np.maximum(X_val_raw, 0.0))

    rss_mean = X_train_log.mean(axis=0).astype(np.float32)
    rss_std = (X_train_log.std(axis=0) + eps).astype(np.float32)

    X_train = ((X_train_log - rss_mean) / rss_std).astype(np.float32)
    X_val = ((X_val_log - rss_mean) / rss_std).astype(np.float32)

    target_min_cm = y_train_cm.min(axis=0).astype(np.float32)
    target_max_cm = y_train_cm.max(axis=0).astype(np.float32)
    target_range_cm = (target_max_cm - target_min_cm + eps).astype(np.float32)

    y_train = ((y_train_cm - target_min_cm) / target_range_cm).astype(np.float32)
    y_val = ((y_val_cm - target_min_cm) / target_range_cm).astype(np.float32)

    model = BaselineMLP(36)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    batch_size = 128
    epochs = 300

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train_t))

        total_loss = 0.0
        for start in range(0, len(X_train_t), batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} train_loss={total_loss / len(X_train_t):.6f} val_loss={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_norm = model(X_val_t).numpy()

    pred_cm = target_min_cm + pred_norm * target_range_cm
    errors = euclidean_errors_cm(pred_cm, y_val_cm)

    print(f"Validation mean error:   {errors.mean():.3f} cm")
    print(f"Validation median error: {np.median(errors):.3f} cm")
    print(f"Validation p95 error:    {np.percentile(errors, 95):.3f} cm")

    torch.save(model.state_dict(), models_dir / "task3_model.pt")

    np.savez(
        models_dir / "task3_scaling.npz",
        rss_mean=rss_mean.astype(np.float32),
        rss_std=rss_std.astype(np.float32),
        target_min_cm=target_min_cm.astype(np.float32),
        target_range_cm=target_range_cm.astype(np.float32),
    )

    print("Saved", models_dir / "task3_model.pt")
    print("Saved", models_dir / "task3_scaling.npz")


if __name__ == "__main__":
    main()
