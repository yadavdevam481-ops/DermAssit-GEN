from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import DermoscopyDataset, CLASS_NAMES
from src.models.classifier import EfficientNetClassifier
from src.models.losses import FocalLoss
from src.utils.metrics import classification_metrics


def make_loader(args, split, train=False):
    ds = DermoscopyDataset(
        args.index, split=split, image_size=args.image_size, train=train,
        synthetic_root=args.synthetic_root, synthetic_multiplier=args.synthetic_multiplier if train else 0
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def class_weights(index_csv):
    import pandas as pd
    df = pd.read_csv(index_csv)
    tr = df[df["split"] == "train"]
    counts = np.bincount(tr["label"].to_numpy(), minlength=len(CLASS_NAMES))
    weights = len(tr) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ys, ps = [], []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y.size(0)
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(torch.softmax(logits, 1).detach().cpu().numpy())
    return loss_sum / total, np.array(ys), np.array(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--synthetic-root", default=None)
    p.add_argument("--synthetic-multiplier", type=int, default=0)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="checkpoints/classifier")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args, args.split, True)
    val_loader = make_loader(args, args.val_split, False)

    model = EfficientNetClassifier(7, pretrained=True).to(device)
    w = class_weights(args.index).to(device)

    if args.loss == "weighted_ce":
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    elif args.loss == "focal":
        loss_fn = FocalLoss(gamma=2.0, weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = -1.0
    patience = 0
    history = []

    for epoch in range(args.epochs):
        tr_loss, tr_y, tr_probs = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_y, va_probs = run_epoch(model, val_loader, loss_fn, device)
        va_pred = va_probs.argmax(1)
        metrics, report = classification_metrics(va_y, va_pred, va_probs, CLASS_NAMES)
        metrics.update({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss})
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        score = metrics["macro_f1"]
        if score > best:
            best = score
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": args.image_size,
                "args": vars(args),
            }, out / "best.pt")
        else:
            patience += 1
        scheduler.step()

        if patience >= 4:
            break

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
