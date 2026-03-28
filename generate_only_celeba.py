import argparse
import csv
import json
import os
import random
import re
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# REM submodule imports
import utils
import attacks
import models


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
    def __init__(self, samples: List[Tuple[str, int, int, str]], input_size: int = 112):
        self.samples = samples
        self.input_size = input_size

        # Base preprocessing: deterministic resize-only 
        self.base_tf = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label, idx, _poisoned_rel_path = self.samples[i]
        img = Image.open(path).convert("RGB")
        x01 = self.base_tf(img)       # [0,1]
        x255 = x01 * 255.0            # [0,255]
        return x255, int(label), int(idx), path


def resolve_poisoned_output_path(out_images_dir: str, clean_path: str, poisoned_rel_path: str, image_format: str) -> str:
    ext = ".png" if image_format == "png" else ".jpg"
    fallback_name = os.path.splitext(os.path.basename(clean_path))[0] + ext

    rel = (poisoned_rel_path or "").strip()
    if not rel:
        rel = fallback_name
    else:
        rel = os.path.normpath(rel)
        if os.path.isabs(rel):
            raise RuntimeError(f"poisoned_rel_path must be relative, got absolute path: {poisoned_rel_path}")
        if rel == ".." or rel.startswith(".." + os.sep):
            raise RuntimeError(f"poisoned_rel_path escapes output directory: {poisoned_rel_path}")

    return os.path.abspath(os.path.join(out_images_dir, rel))


# Endless loader (step-based training like REM's Loader)
class EndlessLoader:
    """Endless iterator over a DataLoader (step-based training like REM scripts)."""
    def __init__(self, dl: DataLoader):
        self.dl = dl
        self.it = None
        self.steps_consumed = 0

    def __len__(self):
        return len(self.dl)

    def __next__(self):
        if self.it is None:
            self.it = iter(self.dl)
        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.dl)
            batch = next(self.it)
        self.steps_consumed += 1
        return batch

    def fast_forward(self, steps: int):
        for _ in range(int(steps)):
            _ = next(self)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def build_train_transforms(input_size: int, enable_eot_aug: bool) -> torch.nn.Module:
    """
    mimic utils.get_transforms(..., is_tensor=True) behavior:
      - expects x in [0,255]
      - normalizes to (x - 127.5)/255 -> approx [-0.5,0.5]

    For train=True, REM adds RandomHorizontalFlip + RandomCrop for CIFAR/tiny-imagenet.
    For CelebA we add:
      - rand flip
      - padded crop to 112 (translation jitter)
      - Open for extensibility
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


def save_uint8_hwc_as_image(arr_hwc: np.ndarray, out_path: str, image_format: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if image_format == "png":
        Image.fromarray(arr_hwc).save(out_path, format="PNG", compress_level=0)
    elif image_format == "jpg":
        Image.fromarray(arr_hwc).save(out_path, format="JPEG", quality=100)
    else:
        raise ValueError(f"Unsupported image_format={image_format}")


def atomic_save_torch(obj: dict, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_csv(path: str, header: List[str], rows: List[List[str]]):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    pat = re.compile(r"^ckpt_step_(\d{8})\.pt$")
    best_step = -1
    best_path = None
    for fn in os.listdir(ckpt_dir):
        m = pat.match(fn)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = os.path.join(ckpt_dir, fn)
    return best_path


def prune_old_checkpoints(ckpt_dir: str, keep_last_k: int):
    if keep_last_k <= 0:
        return
    pat = re.compile(r"^ckpt_step_(\d{8})\.pt$")
    found = []
    for fn in os.listdir(ckpt_dir):
        m = pat.match(fn)
        if m:
            found.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    found.sort(key=lambda x: x[0])
    if len(found) <= keep_last_k:
        return
    to_delete = found[:-keep_last_k]
    for _step, path in to_delete:
        try:
            os.remove(path)
        except Exception:
            pass


def append_jsonl(path: str, rec: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--samples-csv", type=str, required=True)
    p.add_argument("--out-images-dir", type=str, required=True)
    p.add_argument("--out-poison-map", type=str, required=True)
    p.add_argument("--out-metrics-json", type=str, required=True)

    # Core experiment params
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

    # Defender hyperparams
    p.add_argument("--pgd-radius", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--pgd-step-size", type=float, default=1.6)
    p.add_argument("--pgd-random-start", action="store_true")

    # Inner attacker
    p.add_argument("--atk-pgd-radius", type=float, default=4.0)
    p.add_argument("--atk-pgd-steps", type=int, default=10)
    p.add_argument("--atk-pgd-step-size", type=float, default=0.8)
    p.add_argument("--atk-pgd-random-start", action="store_true")

    # EOT
    p.add_argument("--samp-num", type=int, default=5)
    p.add_argument("--no-eot-aug", action="store_true")

    # Frequencies
    p.add_argument("--perturb-freq", type=int, default=1)
    p.add_argument("--report-freq", type=int, default=1000)

    # Runtime
    p.add_argument("--input-size", type=int, default=112)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)

    # Output format
    p.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])

    # Checkpointing + JSONL logging
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default="none",
                   help="none | latest | /abs/path/to/ckpt_step_XXXXXXXX.pt")
    p.add_argument("--keep-last-k", type=int, default=2)
    p.add_argument("--log-jsonl", type=str, required=True)

    args = p.parse_args()

    # Determinism (reasonable baseline; REM itself doesn't fully freeze RNG because aug is stochastic by design)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.out_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_poison_map), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics_json), exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_jsonl), exist_ok=True)

    # Read samples
    samples: List[Tuple[str, int, int, str]] = []
    with open(args.samples_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            clean_path = os.path.abspath(row["clean_path"])
            label = int(row["label"])
            idx = int(row["idx"])
            poisoned_rel_path = row.get("poisoned_rel_path", "")
            samples.append((clean_path, label, idx, poisoned_rel_path))

    if not samples:
        raise RuntimeError("No samples found in samples_csv.")

    n = len(samples)
    idxs = [ii for _, _, ii, _ in samples]
    if sorted(idxs) != list(range(n)):
        raise RuntimeError(
            "samples_csv idx must be a permutation of 0..N-1 "
            f"(min={min(idxs)} max={max(idxs)} unique={len(set(idxs))} N={n})"
        )

    labels = [y for _, y, _, _ in samples]
    num_classes = max(labels) + 1

    # Dataset + loader
    ds = CSVIndexedImageDataset(samples, input_size=args.input_size)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(args.seed)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,   # REM-style training uses shuffle=True
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )
    train_loader = EndlessLoader(dl)

    # Build surrogate model / optim / loss
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

    # Defender + attacker
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
    def_noise = np.zeros([n, 3, args.input_size, args.input_size], dtype=np.int8)

    def save_checkpoint(next_step: int):
        ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_step_{next_step:08d}.pt")
        # Store def_noise as torch int8 tensor (compact, torch-native)
        def_noise_t = torch.from_numpy(def_noise.copy()).to(torch.int8)
        ckpt = {
            "next_step": int(next_step),
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "py_rng": random.getstate(),
            "torch_rng": torch.random.get_rng_state(),
            "np_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "def_noise": def_noise_t,
            "shape": (n, 3, args.input_size, args.input_size),
            "args": vars(args),
        }
        atomic_save_torch(ckpt, ckpt_path)
        prune_old_checkpoints(args.ckpt_dir, args.keep_last_k)
        return ckpt_path

    def load_checkpoint(path: str) -> int:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if ckpt.get("py_rng") is not None:
            random.setstate(ckpt["py_rng"])
        torch.random.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["np_rng"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

        if tuple(ckpt["shape"]) != (n, 3, args.input_size, args.input_size):
            raise RuntimeError(
                f"Checkpoint shape mismatch: ckpt={ckpt['shape']} now={(n,3,args.input_size,args.input_size)}"
            )

        dn = ckpt["def_noise"]
        if isinstance(dn, torch.Tensor):
            dn_np = dn.to(torch.int8).cpu().numpy()
        else:
            dn_np = np.array(dn, dtype=np.int8)
        if dn_np.shape != def_noise.shape:
            raise RuntimeError(f"def_noise shape mismatch: ckpt={dn_np.shape} now={def_noise.shape}")
        def_noise[:] = dn_np
        return int(ckpt["next_step"])

    # Resume
    start_step = 0
    if args.resume != "none":
        if args.resume == "latest":
            latest = find_latest_checkpoint(args.ckpt_dir)
            if latest is None:
                raise RuntimeError(f"--resume latest set but no checkpoints in {args.ckpt_dir}")
            ckpt_path = latest
        else:
            ckpt_path = args.resume

        start_step = load_checkpoint(ckpt_path)
        if start_step > 0:
            print(f"[REM] Fast-forwarding loader by {start_step} steps for deterministic resume...")
            train_loader.fast_forward(start_step)
        print(f"[REM] Resumed from {ckpt_path} at next_step={start_step}")

    t0 = time.perf_counter()
    last_def_acc = None
    last_def_loss = None

    for step in tqdm(
        range(start_step, args.train_steps),
        total=args.train_steps,
        initial=start_step,
        desc="REM train",
        unit="step",
        mininterval=0.5,
    ):
        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group["lr"] = lr

        x, y, ii, _paths = next(train_loader)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ii_np = np.array(ii, dtype=np.int64)

        # Update noise
        if (step + 1) % args.perturb_freq == 0:
            delta = defender.perturb(model, criterion, x, y)
            def_noise[ii_np] = (delta.detach().cpu().numpy() * 255.0).round().astype(np.int8)

        # Apply defended noise and transforms
        noise_t = torch.tensor(def_noise[ii_np], device=device, dtype=torch.float32)
        def_x = train_trans(x + noise_t)
        def_x.clamp_(-0.5, 0.5)

        adv_x = attacker.perturb(model, criterion, def_x, y)

        model.train()
        logits = model(adv_x)
        loss = criterion(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        acc = (logits.argmax(dim=1) == y).float().mean().item()
        last_def_acc = float(acc)
        last_def_loss = float(loss.item())

        if (step + 1) % args.report_freq == 0 or (step + 1) == args.train_steps:
            rec = {
                "step": int(step + 1),
                "lr": float(lr),
                "def_acc": float(acc),
                "def_loss": float(loss.item()),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
            append_jsonl(args.log_jsonl, rec)
            print(f"[REM] step {step+1}/{args.train_steps} | def_acc={acc:.4f} | def_loss={loss.item():.4e}")

        if (step + 1) % args.save_freq == 0:
            ckpt_path = save_checkpoint(step + 1)
            print(f"[REM] Saved checkpoint: {ckpt_path}")

    # Regenerate final noise (one full pass)
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

    # Final checkpoint snapshot (optional but useful)
    fin_ckpt = save_checkpoint(args.train_steps)
    print(f"[REM] Saved final checkpoint: {fin_ckpt}")

    t_total = time.perf_counter() - t0

    # Write poisoned images + poison_map (atomic per-file write)
    poison_rows: List[List[str]] = []
    write_desc = "Write PNGs" if args.image_format == "png" else "Write JPGs"

    seen_poisoned_paths = set()
    for clean_path, _label, idx, poisoned_rel_path in tqdm(samples, desc=write_desc):
        img = Image.open(clean_path).convert("RGB")
        img = img.resize((args.input_size, args.input_size), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.int16)

        noise = def_noise[idx].astype(np.int16)          # CHW
        noise_hwc = np.transpose(noise, (1, 2, 0))       # HWC
        poisoned = np.clip(arr + noise_hwc, 0, 255).astype(np.uint8)

        poisoned_path = resolve_poisoned_output_path(
            out_images_dir=args.out_images_dir,
            clean_path=clean_path,
            poisoned_rel_path=poisoned_rel_path,
            image_format=args.image_format,
        )

        if poisoned_path in seen_poisoned_paths:
            raise RuntimeError(
                f"Duplicate poisoned output path detected: {poisoned_path}. "
                "Regenerate with unique per-sample relative paths."
            )
        seen_poisoned_paths.add(poisoned_path)

        os.makedirs(os.path.dirname(poisoned_path), exist_ok=True)

        tmp_img = poisoned_path + ".tmp"
        save_uint8_hwc_as_image(poisoned, tmp_img, image_format=args.image_format)
        os.replace(tmp_img, poisoned_path)

        poison_rows.append([clean_path, poisoned_path])

    atomic_write_csv(args.out_poison_map, ["clean_path", "poisoned_path"], poison_rows)

    metrics = {
        "method": "REM",
        "total_images": len(samples),
        "total_time_sec": float(t_total),
        "throughput_img_per_sec": (len(samples) / t_total) if t_total > 0 else None,
        "args": vars(args),
        "last_def_acc": last_def_acc,
        "last_def_loss": last_def_loss,
        "log_jsonl": os.path.abspath(args.log_jsonl),
        "ckpt_dir": os.path.abspath(args.ckpt_dir),
    }
    atomic_write_json(args.out_metrics_json, metrics)

    print(f"[REM] Saved poisoned images: {args.out_images_dir}")
    print(f"[REM] Saved poison map:      {args.out_poison_map}")
    print(f"[REM] Saved metrics:        {args.out_metrics_json}")
    print(f"[REM] Log JSONL:            {args.log_jsonl}")
    print(f"[REM] Checkpoints:          {args.ckpt_dir}")


if __name__ == "__main__":
    main()
