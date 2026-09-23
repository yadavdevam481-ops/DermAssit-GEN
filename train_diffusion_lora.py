from __future__ import annotations

"""
Explicit Stable-Diffusion v1.x + LoRA training loop.

The implementation follows the standard Diffusers text-to-image LoRA pattern:
freeze the pretrained VAE/text encoder/UNet weights, insert LoRA adapters into
UNet attention projections, and optimize only the adapter parameters.

For this project, the text prompt is derived from the HAM10000 diagnostic label.
This is a research adaptation, not a clinically validated generative model.
"""

import argparse
import json
from pathlib import Path
import math
import random

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers


PROMPTS = {
    "akiec": "a dermoscopic image of actinic keratoses, clinical dermatology research image",
    "bcc": "a dermoscopic image of basal cell carcinoma, clinical dermatology research image",
    "bkl": "a dermoscopic image of benign keratosis, clinical dermatology research image",
    "df": "a dermoscopic image of dermatofibroma, clinical dermatology research image",
    "mel": "a dermoscopic image of melanoma, clinical dermatology research image",
    "nv": "a dermoscopic image of melanocytic nevi, clinical dermatology research image",
    "vasc": "a dermoscopic image of vascular lesions, clinical dermatology research image",
}


class DiffusionDataset(Dataset):
    def __init__(self, index_csv, split, resolution, classes=None):
        df = pd.read_csv(index_csv)
        df = df[df["split"] == split].copy()
        if classes:
            df = df[df["dx"].isin(classes)].copy()
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        image = self.transform(Image.open(r["image_path"]).convert("RGB"))
        return {"pixel_values": image, "prompt": PROMPTS[r["dx"]], "class_name": r["dx"]}


def tokenize_prompts(tokenizer, prompts):
    ids = tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return ids


def add_lora_to_unet(unet, rank):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(config)
    for name, p in unet.named_parameters():
        p.requires_grad = "lora_" in name


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    add_lora_to_unet(unet, args.rank)

    dataset = DiffusionDataset(
        args.index, args.split, args.resolution,
        classes=args.classes if args.classes else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=1e-2)

    unet, text_encoder, vae, optimizer, loader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, loader
    )

    global_step = 0
    progress = tqdm(total=args.steps, disable=not accelerator.is_local_main_process)

    epoch = 0
    while global_step < args.steps:
        epoch += 1
        for batch in loader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                input_ids = tokenize_prompts(tokenizer, batch["prompt"])
                input_ids = input_ids.to(latents.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach().item()))

                if global_step % args.save_every == 0:
                    if accelerator.is_main_process:
                        raw_unet = accelerator.unwrap_model(unet).to(torch.float32)
                        lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(raw_unet))
                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=str(output / f"checkpoint-{global_step}"),
                            unet_lora_layers=lora_state,
                            safe_serialization=True,
                        )

            if global_step >= args.steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw_unet = accelerator.unwrap_model(unet).to(torch.float32)
        lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(raw_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output),
            unet_lora_layers=lora_state,
            safe_serialization=True,
        )
        with open(output / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    accelerator.end_training()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="checkpoints/diffusion_lora")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
