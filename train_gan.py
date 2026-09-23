from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, utils
from src.models.cgan import ConditionalGenerator, ConditionalDiscriminator


class CSVDataset(Dataset):
    def __init__(self, csv, split="train", image_size=128):
        df = pd.read_csv(csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.t(Image.open(row["image_path"]).convert("RGB"))
        return x, int(row["label"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--output-dir", default="checkpoints/cgan")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    ds = CSVDataset(args.index, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=True)

    G = ConditionalGenerator(z_dim=args.z_dim).to(device)
    D = ConditionalDiscriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_z = torch.randn(14, args.z_dim, device=device)
    fixed_y = torch.tensor([3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 5, 5], device=device)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        g_sum = d_sum = 0.0
        for real, labels in dl:
            real, labels = real.to(device), labels.to(device)
            b = real.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # D
            opt_d.zero_grad(set_to_none=True)
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z, labels).detach()
            d_loss = criterion(D(real, labels), ones) + criterion(D(fake, labels), zeros)
            d_loss.backward()
            opt_d.step()

            # G
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(b, args.z_dim, device=device)
            fake = G(z, labels)
            g_loss = criterion(D(fake, labels), ones)
            g_loss.backward()
            opt_g.step()

            g_sum += g_loss.item()
            d_sum += d_loss.item()

        with torch.no_grad():
            grid = G(fixed_z, fixed_y).cpu()
            utils.save_image(grid, out / "samples" / f"epoch_{epoch:04d}.png", nrow=7, normalize=True)

        torch.save({"generator": G.state_dict(), "z_dim": args.z_dim}, out / "last.pt")
        if g_sum < best:
            best = g_sum
            torch.save({"generator": G.state_dict(), "z_dim": args.z_dim}, out / "best.pt")

        print(f"epoch={epoch} D={d_sum/len(dl):.4f} G={g_sum/len(dl):.4f}")


if __name__ == "__main__":
    main()
