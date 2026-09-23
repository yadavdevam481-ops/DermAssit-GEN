from __future__ import annotations

import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision.utils import save_image
from src.models.cgan import ConditionalGenerator
from src.data.dataset import CLASS_TO_ID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--class", dest="class_name", required=True, choices=list(CLASS_TO_ID))
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--output-dir", default="data/synthetic/cgan")
    p.add_argument("--z-dim", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    G = ConditionalGenerator(z_dim=ckpt.get("z_dim", args.z_dim)).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    out = Path(args.output_dir) / args.class_name
    out.mkdir(parents=True, exist_ok=True)
    label = CLASS_TO_ID[args.class_name]

    with torch.no_grad():
        for start in range(0, args.num_images, 32):
            n = min(32, args.num_images - start)
            z = torch.randn(n, args.z_dim, device=device)
            y = torch.full((n,), label, device=device, dtype=torch.long)
            imgs = G(z, y)
            for i, img in enumerate(imgs):
                save_image(img, out / f"{start+i:05d}.png", normalize=True)

    print("saved", args.num_images, "images to", out)


if __name__ == "__main__":
    main()
