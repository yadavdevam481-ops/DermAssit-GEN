from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DermoscopyDataset, CLASS_NAMES
from src.models.classifier import EfficientNetClassifier
from src.utils.metrics import classification_metrics, save_confusion_matrix, bootstrap_metric_ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = DermoscopyDataset(args.index, split=args.split, image_size=args.image_size, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = EfficientNetClassifier(7, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred, probs = [], [], []
    paths = []

    with torch.no_grad():
        for x, y, pth in dl:
            logits = model(x.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            y_pred.extend(p.argmax(1).tolist())
            y_true.extend(y.tolist())
            paths.extend(pth)

    probs = np.concatenate(probs, axis=0)
    metrics, report = classification_metrics(np.array(y_true), np.array(y_pred), probs, CLASS_NAMES)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score
    ci = {
        "macro_f1": bootstrap_metric_ci(np.array(y_true), np.array(y_pred), lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": bootstrap_metric_ci(np.array(y_true), np.array(y_pred), balanced_accuracy_score),
        "macro_recall": bootstrap_metric_ci(np.array(y_true), np.array(y_pred), lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0)),
    }
    with open(out / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci, f, indent=2)

    with open(out / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, out / "confusion_matrix.png")

    import pandas as pd
    pred_df = pd.DataFrame({
        "image_path": paths,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for i, c in enumerate(CLASS_NAMES):
        pred_df[f"prob_{c}"] = probs[:, i]
    pred_df.to_csv(out / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
