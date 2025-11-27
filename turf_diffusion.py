import os
import argparse
from pathlib import Path

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def build_diffusion_model(image_size: int) -> GaussianDiffusion:
    """
    Create the UNet backbone and wrap it in a GaussianDiffusion object.
    Adjust dim and dim_mults if you have a very small / very large GPU.
    """

    # Trial 1
    trial = 1
    
    if trial == 1 :
        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),  
            channels=3,
        )
    elif trial == 2:
        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),  
            channels=3,
        )
    else:
        raise ValueError(f"Unsupported trial configuration: {trial}")

    diffusion = GaussianDiffusion(
        model=model,
        image_size=image_size,
        timesteps=250,          # number of diffusion steps (DDPM)
        sampling_timesteps=50,  # use fewer steps at sampling time for speed
        objective='pred_noise',  # standard DDPM objective
    )

    return diffusion


def train(
    data_dir: str,
    image_size: int = 256,
    results_dir: str = "./turf_results",
    batch_size: int = 6,
    num_steps: int = 100000,
    lr: float = 1e-4,
    save_every: int = 5000,
    grad_accum: int = 2,
):
    """
    Train a diffusion model on all images in data_dir.

    data_dir should look like:
        data_dir/
            patch_0001.png
            patch_0002.jpg
            ...

    All images will be resized and centre-cropped to image_size.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    diffusion = build_diffusion_model(image_size=image_size)

    trainer = Trainer(
        diffusion,
        data_dir,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_steps=num_steps,
        gradient_accumulate_every=grad_accum,
        ema_decay=0.995,
        amp=True,                          # mixed precision if your GPU supports it
        results_folder=results_dir,
        save_and_sample_every=save_every
    )

    print(f"Starting training on images in: {data_dir}")
    print(f"Results (checkpoints + samples) will be saved to: {results_dir}")
    trainer.train()

def sample(
    ckpt_path: str,
    image_size: int = 256,
    num_samples: int = 100,
    out_dir: str = "./turf_samples",
    max_batch_size: int = 4,       # safe chunk size for a 4080
):
    import math
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    diffusion = build_diffusion_model(image_size=image_size)

    ckpt = torch.load(ckpt_path, map_location=device)

    # correct way for your checkpoint structure
    if isinstance(ckpt, dict) and "model" in ckpt:
        print("Loading diffusion weights from ckpt['model']")
        diffusion.load_state_dict(ckpt["model"])
    else:
        print("Assuming checkpoint is already a diffusion state_dict")
        diffusion.load_state_dict(ckpt)

    diffusion.to(device)
    diffusion.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from torchvision.utils import save_image

    total = num_samples
    saved = 0
    remaining = num_samples

    print(f"📸 Starting sampling: {num_samples} images")
    print(f"⚡ Generating in batches of {max_batch_size}")

    with torch.no_grad():
        while remaining > 0:
            batch_size = min(max_batch_size, remaining)
            print(f"   → Sampling batch {saved // max_batch_size + 1} "
                  f"({batch_size} images)...")

            # Mixed precision helps
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    imgs = diffusion.sample(batch_size=batch_size)
            else:
                imgs = diffusion.sample(batch_size=batch_size)

            # Save each image
            for i in range(batch_size):
                idx = saved + i
                save_path = out_path / f"sample_{idx:06d}.png"  # unlimited numbering
                save_image(imgs[i], save_path)

            saved += batch_size
            remaining -= batch_size

            print(f" ✓ Saved {saved}/{total} images")

            # Free VRAM
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n🎉 Sampling complete.")
    print(f"Images stored in: {out_path}")




def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or sample a diffusion model on turfgrass images using denoising-diffusion-pytorch."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---- Train subcommand ----
    train_parser = subparsers.add_parser("train", help="Train a diffusion model")
    train_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder containing training images (turfgrass patches).",
    )
    train_parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size (images will be resized/cropped to this).",
    )
    train_parser.add_argument(
        "--results_dir",
        type=str,
        default="./turf_results",
        help="Where to store checkpoints and sample grids.",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    train_parser.add_argument(
        "--num_steps",
        type=int,
        default=100_000,
        help="Total training steps.",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    train_parser.add_argument(
        "--save_every",
        type=int,
        default=5_000,
        help="Save & sample every N training steps.",
    )
    train_parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (acts like larger batch size).",
    )

    # ---- Sample subcommand ----
    sample_parser = subparsers.add_parser("sample", help="Sample from a trained diffusion model")
    sample_parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a trained checkpoint (e.g., ./turf_results/ema-100000.pt).",
    )
    sample_parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size that the model was trained on.",
    )
    sample_parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of images to sample.",
    )
    sample_parser.add_argument(
        "--out_dir",
        type=str,
        default="./turf_samples",
        help="Folder to save sampled images.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train(
            data_dir=args.data_dir,
            image_size=args.image_size,
            results_dir=args.results_dir,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            lr=args.lr,
            save_every=args.save_every,
            grad_accum=args.grad_accum,
        )
    elif args.mode == "sample":
        sample(
            ckpt_path=args.ckpt,
            image_size=args.image_size,
            num_samples=args.num_samples,
            out_dir=args.out_dir,
        )


# RUN 1 
# python turf_diffusion.py train  --data_dir "C:\data\turfgrass\train\good"  --image_size 256  --results_dir "C:\Users\stevf\OneDrive\Documents\Projects\denoising-diffusion-pytorch\results"  --batch_size 8 --grad_accum 2  --num_steps 100000

#  python turf_diffusion.py sample  --ckpt "C:\Users\stevf\OneDrive\Documents\Projects\denoising-diffusion-pytorch\results_32\model-2.pt"  --image_size 256  --num_samples 100  --out_dir "C:\Users\stevf\OneDrive\Documents\Projects\denoising-diffusion-pytorch\results_32\turf_samples"
