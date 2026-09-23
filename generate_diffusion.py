from __future__ import annotations

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig
from src.data.dataset import CLASS_NAMES


PROMPTS = {
    "akiec": "a dermoscopic image of actinic keratoses, clinical dermatology research image",
    "bcc": "a dermoscopic image of basal cell carcinoma, clinical dermatology research image",
    "bkl": "a dermoscopic image of benign keratosis, clinical dermatology research image",
    "df": "a dermoscopic image of dermatofibroma, clinical dermatology research image",
    "mel": "a dermoscopic image of melanoma, clinical dermatology research image",
    "nv": "a dermoscopic image of melanocytic nevi, clinical dermatology research image",
    "vasc": "a dermoscopic image of vascular lesions, clinical dermatology research image",
}


def load_lora(pipe, lora_dir):
    pipe.load_lora_weights(str(lora_dir), weight_name="pytorch_lora_weights.safetensors")
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--class", dest="class_name", required=True, choices=CLASS_NAMES)
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--output-dir", default="data/synthetic/diffusion")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    pipe = load_lora(pipe, args.lora_dir)

    out = Path(args.output_dir) / args.class_name
    out.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for i in range(args.num_images):
        image = pipe(
            PROMPTS[args.class_name],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        image.save(out / f"{i:05d}.png")

    print("saved", args.num_images, "images to", out)


if __name__ == "__main__":
    main()
