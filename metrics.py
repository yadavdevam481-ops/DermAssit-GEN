from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true, y_pred, probs=None, class_names=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if probs is not None:
        y_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
        try:
            out["macro_roc_auc_ovr"] = float(
                roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
            )
        except ValueError:
            out["macro_roc_auc_ovr"] = None

    report = classification_report(
        y_true, y_pred,
        labels=np.arange(len(class_names)) if class_names else None,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return out, report


def save_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, seed=42, alpha=0.05):
    """Non-parametric bootstrap percentile confidence interval."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    values=[]; n=len(y_true)
    for _ in range(n_bootstrap):
        idx=rng.integers(0,n,size=n)
        try: values.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception: pass
    if not values: raise RuntimeError('Bootstrap produced no valid samples.')
    lo,hi=np.quantile(values,[alpha/2,1-alpha/2])
    return {'estimate':float(metric_fn(y_true,y_pred)),'lower':float(lo),'upper':float(hi),'n_bootstrap':len(values),'confidence_level':1-alpha}
