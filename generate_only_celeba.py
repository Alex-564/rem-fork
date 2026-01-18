import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# REM submodule imports 
import utils
import attacks
import models


# Dataset: manifest-driven CSV
class CSVIndexedImageDataset(Dataset):
    """
    Reads a samples CSV with columns:
      - clean_path: path to image (string)
      - label: int label in [0..C-1]
      - idx: stable integer sample index (used to index def_noise)

    Returns:
      x: float tensor in [0,255], shape [3,112,112]
      y: int64 tensor
      ii: int64 tensor (stable)
      path: str (for later writing poison_map)
    """
    def __init__(self, samples: List[Tuple[str, int, int]], input_size: int = 112):
        self.samples = samples
        self.input_size = input_size

        # Base preprocessing: deterministic resize-only 
        self.base_tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label, idx = self.samples[i]
        img = Image.open(path).convert("RGB")
        x01 = self.base_tf(img)           # [0,1]
        x255 = x01 * 255.0                # [0,255] float
        return x255, int(label), int(idx), path


# Endless loader (step-based training like REM's Loader)
class EndlessLoader:
    def __init__(self, dl: DataLoader):
        self.dl = dl
        self.it = None

    def __len__(self):
        return len(self.dl)

    def __next__(self):
        if self.it is None:
            self.it = iter(self.dl)
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.dl)
            return next(self.it)


# CelebA specific model builder 
def get_arch_celeba(arch: str, num_classes: int):
    in_dims = 3
    out_dims = num_classes

    if arch == "resnet18":
        return models.resnet18(in_dims, out_dims)
    elif arch == "resnet50":
        return models.resnet50(in_dims, out_dims)
    elif arch == "wrn-34-10":
        return models.wrn34_10(in_dims, out_dims)
    elif arch == "vgg11-bn":
        return models.vgg11_bn(in_dims, out_dims)
    elif arch == "vgg16-bn":
        return models.vgg16_bn(in_dims, out_dims)
    elif arch == "vgg19-bn":
        return models.vgg19_bn(in_dims, out_dims)
    elif arch == "densenet-121":
        return models.densenet121(num_classes=out_dims)
    else:
        raise NotImplementedError(f"Unsupported arch={arch}")


# Transforms: REM tensor-domain normalization + optional EOT aug
def build_train_transforms(
    input_size: int,
    enable_eot_aug: bool,
) -> torch.nn.Module:
    """
    mimic utils.get_transforms(..., is_tensor=True) behavior:
      - expects x in [0,255]
      - normalizes to (x - 127.5)/255 -> approx [-0.5,0.5]

    For train=True, REM adds RandomHorizontalFlip + RandomCrop for CIFAR/tiny-imagenet.
    For CelebA we add:
      - rand flip
      - padded crop to 112 (translation jitter)
      - Open for extnesibiliy
    NOTE: Any transforms here directly affect the EOT "robustness" of the generated poison. Affects ablation
    """
    comp1 = []
    if enable_eot_aug:
        ## TODO: ADD CUSTOM AUGMENTATIONS HERE ##
        comp1 = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_size, padding=8),
        ]

    # Standard REM normalization to approx [-0.5,0.5]
    comp2 = [
        transforms.Normalize((255 * 0.5, 255 * 0.5, 255 * 0.5), (255.0, 255.0, 255.0))
    ]

    trans = transforms.Compose([*comp1, *comp2])

    # REM wraps tensor-mode transforms with ElementWiseTransform for batched tensors. 
    return utils.data.ElementWiseTransform(trans)


# Utility: write JPG from uint8 HWC
def save_uint8_hwc_as_jpg(arr_hwc: np.ndarray, out_path: str, quality: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr_hwc).save(out_path, quality=quality)


# Main generation logic (mirrors generate_robust_em.py)
def main():
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--samples-csv", type=str, required=True)
    p.add_argument("--out-images-dir", type=str, required=True)
    p.add_argument("--out-poison-map", type=str, required=True)
    p.add_argument("--out-metrics-json", type=str, required=True)

    # Core experiment params (mirrors shared args)
    p.add_argument("--arch", type=str, default="resnet18",
                   choices=["resnet18", "resnet50", "vgg11-bn", "vgg16-bn", "vgg19-bn", "densenet-121", "wrn-34-10"])
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-decay-rate", type=float, default=0.1)
    p.add_argument("--lr-decay-freq", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)

    # Defender (REM) hyperparams (same names as paper scripts)
    p.add_argument("--pgd-radius", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--pgd-step-size", type=float, default=1.6)
    p.add_argument("--pgd-random-start", action="store_true")

    # Inner attacker for minimax (as in generate_robust_em.py)
    p.add_argument("--atk-pgd-radius", type=float, default=4.0)
    p.add_argument("--atk-pgd-steps", type=int, default=10)
    p.add_argument("--atk-pgd-step-size", type=float, default=0.8)
    p.add_argument("--atk-pgd-random-start", action="store_true")

    # EOT
    p.add_argument("--samp-num", type=int, default=5)
    p.add_argument("--no-eot-aug", action="store_true",
                   help="Disable stochastic flip/crop augmentations in train_trans (still uses samp_num loop).")

    # Frequencies
    p.add_argument("--perturb-freq", type=int, default=1)
    p.add_argument("--report-freq", type=int, default=1000)

    # Runtime
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)

    # Output fidelity
    p.add_argument("--save-quality", type=int, default=100)

    args = p.parse_args()

    # Determinism (reasonable baseline; REM itself doesn't fully freeze RNG because aug is stochastic by design)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_poison_map), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics_json), exist_ok=True)

    # Read samples CSV (manifest)
    samples: List[Tuple[str, int, int]] = []
    with open(args.samples_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            clean_path = os.path.abspath(row["clean_path"])
            label = int(row["label"])
            idx = int(row["idx"])
            samples.append((clean_path, label, idx))

    if not samples:
        raise RuntimeError("No samples found in samples_csv.")

    # Determine num_classes
    labels = [y for _, y, _ in samples]
    num_classes = max(labels) + 1

    # Dataset + loader
    ds = CSVIndexedImageDataset(samples, input_size=args.input_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,           # REM generation trains with shuffle=True, drop_last=True for train loader 
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader = EndlessLoader(dl)

    # Build surrogate model / optim / loss (Same as REM just pulled out functionality)
    model = get_arch_celeba(args.arch, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    if args.optim == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        optim = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    device = torch.device(args.device)
    model = model.to(device)
    criterion = criterion.to(device)

    # Train transforms (EOT distribution)
    train_trans = build_train_transforms(
        input_size=args.input_size,
        enable_eot_aug=(not args.no_eot_aug),
    )

    # Defender + attacker (mirrors generate_robust_em.py)
    defender = attacks.RobustMinimaxPGDDefender(
        samp_num=args.samp_num,
        trans=train_trans,
        radius=args.pgd_radius,
        steps=args.pgd_steps,
        step_size=args.pgd_step_size,
        random_start=args.pgd_random_start,
        atk_radius=args.atk_pgd_radius,
        atk_steps=args.atk_pgd_steps,
        atk_step_size=args.atk_pgd_step_size,
        atk_random_start=args.atk_pgd_random_start,
    )

    attacker = attacks.PGDAttacker(
        radius=args.atk_pgd_radius,
        steps=args.atk_pgd_steps,
        step_size=args.atk_pgd_step_size,
        random_start=args.atk_pgd_random_start,
        norm_type="l-infty",
        ascending=True,
    )

    # def_noise buffer: int8 in pixel units, indexed by idx (ii)
    # This mirrors REM storing (delta*255).round().astype(np.int8). 
    data_nums = len(ds)
    def_noise = np.zeros([data_nums, 3, args.input_size, args.input_size], dtype=np.int8) # initalise noise for celebA



    # Training loop (step-based like original)
    log: Dict[str, List[float]] = {}
    t0 = time.perf_counter()

    for step in range(args.train_steps):
        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group["lr"] = lr

        x, y, ii, _paths = next(train_loader)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ii_np = np.array(ii, dtype=np.int64)

        # Update noise
        if (step + 1) % args.perturb_freq == 0:
            delta = defender.perturb(model, criterion, x, y)  # delta in normalized units (approx) but returned as float
            def_noise[ii_np] = (delta.detach().cpu().numpy() * 255.0).round().astype(np.int8)

        # Apply defended noise and transforms
        noise_t = torch.tensor(def_noise[ii_np], device=device, dtype=torch.float32)
        def_x = train_trans(x + noise_t)
        def_x.clamp_(-0.5, 0.5)  # original script clamps def_x in normalized domain :contentReference[oaicite:17]{index=17}

        # Inner attacker (adversarially trained learner simulation)
        adv_x = attacker.perturb(model, criterion, def_x, y)

        # ERM step on adv_x
        model.train()
        _y = model(adv_x)
        loss = criterion(_y, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        acc = (_y.argmax(dim=1) == y).float().mean().item()
        utils.add_log(log, "def_acc", acc)
        utils.add_log(log, "def_loss", float(loss.item()))

        if (step + 1) % args.report_freq == 0:
            print(f"[REM] step {step+1}/{args.train_steps} | def_acc={acc:.4f} | def_loss={loss.item():.4e}")

    # Regenerate final noise (original script does this after training) 
    print("[REM] Regenerating final defensive noise (one full pass)...")
    regen_dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    for x, y, ii, _paths in tqdm(regen_dl, desc="REM regen"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ii_np = np.array(ii, dtype=np.int64)
        delta = defender.perturb(model, criterion, x, y)
        def_noise[ii_np] = (delta.detach().cpu().numpy() * 255.0).round().astype(np.int8)

    t_total = time.perf_counter() - t0

    # Write poisoned images + poison_map (clean_path -> poisoned_path)
    poison_map: Dict[str, str] = {}
    for clean_path, label, idx in tqdm(samples, desc="Write JPGs"):
        # Load clean again (same base preprocessing)
        img = Image.open(clean_path).convert("RGB")
        img = img.resize((args.input_size, args.input_size), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.int16)  # HWC 0..255

        noise = def_noise[idx].astype(np.int16) # CHW
        noise_hwc = np.transpose(noise, (1, 2, 0)) # HWC

        poisoned = np.clip(arr + noise_hwc, 0, 255).astype(np.uint8)

        fname = os.path.basename(clean_path)
        poisoned_path = os.path.abspath(os.path.join(args.out_images_dir, fname))
        save_uint8_hwc_as_jpg(poisoned, poisoned_path, quality=args.save_quality)

        poison_map[clean_path] = poisoned_path

    # poison_map.csv
    with open(args.out_poison_map, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clean_path", "poisoned_path"])
        for k, v in poison_map.items():
            w.writerow([k, v])

    # metrics.json
    metrics = {
        "method": "REM",
        "total_images": len(samples),
        "total_time_sec": t_total,
        "throughput_img_per_sec": (len(samples) / t_total) if t_total > 0 else None,
        "args": vars(args),
        "log_keys": list(log.keys()),
        "last_def_acc": log["def_acc"][-1] if "def_acc" in log and log["def_acc"] else None,
        "last_def_loss": log["def_loss"][-1] if "def_loss" in log and log["def_loss"] else None,
    }
    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[REM] Saved poisoned images: {args.out_images_dir}")
    print(f"[REM] Saved poison map:      {args.out_poison_map}")
    print(f"[REM] Saved metrics:        {args.out_metrics_json}")


if __name__ == "__main__":
    main()
