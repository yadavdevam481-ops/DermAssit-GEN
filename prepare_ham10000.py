from __future__ import annotations

import argparse
from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def resolve_image(image_id: str, image_root: Path):
    for ext in (".jpg", ".jpeg", ".png"):
        matches = list(image_root.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    required = {"image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    df = df[df["dx"].isin(CLASSES)].copy()
    df["image_path"] = [
        str(p) if (p := resolve_image(i, Path(args.image_root))) else ""
        for i in df["image_id"].astype(str)
    ]
    df = df[df["image_path"] != ""].copy()
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})

    # Patient-wise splitting when HAM10000 metadata has lesion_id.
    group_col = "lesion_id" if "lesion_id" in df.columns else None
    if group_col:
        groups = df[[group_col, "dx"]].drop_duplicates(group_col)
        train_g, temp_g = train_test_split(
            groups, test_size=0.30, random_state=args.seed, stratify=groups["dx"]
        )
        val_g, test_g = train_test_split(
            temp_g, test_size=0.50, random_state=args.seed, stratify=temp_g["dx"]
        )
        split_map = {g: "train" for g in train_g[group_col]}
        split_map.update({g: "val" for g in val_g[group_col]})
        split_map.update({g: "test" for g in test_g[group_col]})
        df["split"] = df[group_col].map(split_map)
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=args.seed, stratify=df["dx"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=args.seed, stratify=temp_df["dx"]
        )
        train_df = train_df.assign(split="train")
        val_df = val_df.assign(split="val")
        test_df = test_df.assign(split="test")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    keep = ["image_id", "image_path", "dx", "label", "split"] + ([group_col] if group_col else [])
    df[keep].to_csv(out / "index.csv", index=False)

    print("Saved:", out / "index.csv")
    print(df.groupby(["split", "dx"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
