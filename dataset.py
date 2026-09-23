from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}


class DermoscopyDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        split: str = "train",
        image_size: int = 224,
        train: bool = False,
        synthetic_root: Optional[str] = None,
        synthetic_multiplier: int = 0,
    ) -> None:
        self.df = pd.read_csv(index_csv)
        self.df = self.df[self.df["split"] == split].copy()

        real = self.df[["image_path", "label", "split"]].copy()

        if synthetic_root and split == "train" and synthetic_multiplier > 0:
            root = Path(synthetic_root)
            syn_rows = []
            per_class = {}
            for class_name in CLASS_NAMES:
                files = sorted((root / class_name).glob("*.png")) + sorted((root / class_name).glob("*.jpg"))
                if not files:
                    continue
                target_n = len(files) * synthetic_multiplier
                chosen = [files[i % len(files)] for i in range(target_n)]
                for p in chosen:
                    syn_rows.append({"image_path": str(p), "label": CLASS_TO_ID[class_name], "split": "train"})
                per_class[class_name] = target_n
            if syn_rows:
                real = pd.concat([real, pd.DataFrame(syn_rows)], ignore_index=True)

        self.df = real.reset_index(drop=True)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]).convert("RGB") as im:
            image = self.transform(im)
        label = int(row["label"])
        return image, label, row["image_path"]
