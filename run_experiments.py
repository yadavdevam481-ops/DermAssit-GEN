from __future__ import annotations

"""
Run/evaluate the 0x/1x/3x/5x synthetic augmentation study.

This script expects classifier checkpoints to already exist. It produces
a compact experiment table from outputs/<name>/metrics.json.
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--experiments", nargs="+", default=[
        "baseline", "aug_1x", "aug_3x", "aug_5x"
    ])
    p.add_argument("--save", default="outputs/experiment_summary.csv")
    args = p.parse_args()

    rows = []
    for name in args.experiments:
        path = Path(args.output_root) / name / "metrics.json"
        if not path.exists():
            print("missing:", path)
            continue
        data = json.loads(path.read_text())
        data["experiment"] = name
        rows.append(data)

    if not rows:
        raise SystemExit("No metrics files found.")

    df = pd.DataFrame(rows)
    cols = [c for c in [
        "experiment", "accuracy", "balanced_accuracy", "macro_f1",
        "weighted_f1", "macro_precision", "macro_recall", "macro_roc_auc_ovr"
    ] if c in df.columns]
    df[cols].to_csv(args.save, index=False)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
