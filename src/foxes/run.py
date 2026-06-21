"""mini-FOXES training / smoke CLI (Makefile-driven).

Phase 1 wired `--data synthetic`: random `(B, 7, 128, 128)` inputs + random log-SXR targets,
runnable on CPU with no data download and no GPU -- the documented Option-B fallback.

Phase 2 wires `--data foxes`: a *tiny* streaming subsample of the FOXES authors' own published
dataset (`griffingoodwin04/FOXES-Data`) flows through the Phase-1 network + loss on CPU. The
real records are resized 512^2 -> 128^2 and their GOES SXR flux is converted to a normalized
log-space label (see `data.py`). A small `data_smoke.json` artifact records tensor shape /
dtype / value-range / label stats so the `validate-foxes` gate can assert the real path works
without re-downloading.

This pins the public signatures (the "C header" of the showcase) and produces a
provenance-shaped `metrics.json` plus an atomic checkpoint, so the `validate-foxes`
faithfulness gate can run before any cloud money is spent on the full training run (Phase 3).

Provenance follows the repo convention (eval_vision.py / train_lstm.py):
  * every output JSON embeds `"config": vars(args) | {...}`,
  * writes are atomic (temp file + os.replace),
  * the training loop supports `--resume` and `--checkpoint-every` (default 250).

Usage:
    python src/foxes/run.py --data synthetic --epochs 1               # CPU smoke (no data)
    python src/foxes/run.py --data foxes --subsample-n 8 --epochs 1   # tiny real data pull
    python src/foxes/run.py --data synthetic --epochs 5 --resume      # resume from checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from .model import ViTLocal

OUT_DIR = Path("results/foxes_repro")
METRICS_FILE = OUT_DIR / "metrics.json"
CKPT_FILE = OUT_DIR / "checkpoint.pt"
DATA_SMOKE_FILE = OUT_DIR / "data_smoke.json"


def atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON (temp file + os.replace), mirroring the repo's provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def atomic_save_checkpoint(path: Path, state: dict) -> None:
    """Atomically `torch.save` a checkpoint (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_model(args: argparse.Namespace) -> ViTLocal:
    return ViTLocal(
        in_chans=args.in_chans,
        img=args.img,
        patch=args.patch,
        embed=args.embed,
        depth=args.depth,
        heads=args.heads,
        mlp=args.mlp,
        local_window=args.local_window,
        aux_mask_chans=args.aux_mask_chans,
    )


def synthetic_batch(
    args: argparse.Namespace, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random `(B, in_chans, img, img)` inputs + random log-SXR targets (Option-B fallback)."""
    x = torch.randn(args.batch, args.in_chans, args.img, args.img, generator=generator)
    # Normalized log-space SXR targets (stand-in; the real labels are the `--data foxes` path).
    y = torch.randn(args.batch, generator=generator)
    return x, y


def load_foxes_data(args: argparse.Namespace) -> dict:
    """Pull the tiny FOXES-Data subsample and write a `data_smoke.json` provenance artifact.

    Returns the dict from `data.load_foxes_subsample` (train/val tensors + materialization
    metadata). The smoke artifact records exactly the things `validate-foxes` asserts: tensor
    shape, dtype, value range (expected within [-1, 1]+/-eps), finite normalized labels, and
    that streaming bounded the pull to `subsample_n` records.
    """
    from .data import DATASET_REPO, load_foxes_subsample

    bundle = load_foxes_subsample(
        subsample_n=args.subsample_n,
        img=args.img,
        val_frac=args.val_frac,
        streaming=True,
    )
    x = bundle["x_train"]
    y_all = torch.cat([bundle["y_train"], bundle["y_val"]])
    x_all = torch.cat([bundle["x_train"], bundle["x_val"]])
    smoke = {
        "dataset_repo": DATASET_REPO,
        "subsample_n": args.subsample_n,
        "n_materialized": bundle["n_materialized"],
        "is_streaming": bundle["is_streaming"],
        "n_train": int(bundle["x_train"].shape[0]),
        "n_val": int(bundle["x_val"].shape[0]),
        "tensor_shape": list(x.shape),  # (n_train, 7, img, img)
        "dtype": str(x.dtype),
        "value_min": float(x_all.min()),
        "value_max": float(x_all.max()),
        "label_min": float(y_all.min()),
        "label_max": float(y_all.max()),
        "labels_finite": bool(torch.isfinite(y_all).all()),
    }
    atomic_write_json(DATA_SMOKE_FILE, {"config": vars(args), "data_smoke": smoke})
    print(f"wrote {DATA_SMOKE_FILE}")
    print("data_smoke:", json.dumps(smoke, indent=2))
    return bundle


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.HuberLoss(delta=0.3)

    # Phase 2: the real FOXES-Data subsample (loaded once) vs. Phase-1 synthetic random tensors.
    foxes_batch: tuple[torch.Tensor, torch.Tensor] | None = None
    if args.data == "foxes":
        bundle = load_foxes_data(args)
        foxes_batch = (bundle["x_train"], bundle["y_train"])

    start_epoch = 0
    if args.resume and CKPT_FILE.exists():
        ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ckpt_epoch = ckpt.get("epoch")
        start_epoch = int(ckpt_epoch or 0) + 1
        print(f"[resume] loaded epoch {ckpt_epoch}, continuing from {start_epoch}")

    losses: list[float] = []
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if foxes_batch is not None:
            x, y = foxes_batch  # one real forward+loss pass over the subsample (Phase 2 scope)
        else:
            x, y = synthetic_batch(args, gen)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        global_pred, _per_patch = model(x)
        loss = loss_fn(global_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"[epoch {epoch}] loss={losses[-1]:.6f}")

        if args.checkpoint_every and ((epoch + 1) % args.checkpoint_every == 0):
            atomic_save_checkpoint(
                CKPT_FILE, {"model": model.state_dict(), "config": vars(args), "epoch": epoch}
            )

    # Always save a final checkpoint so --resume and the reload assertion have something.
    last_epoch = max(args.epochs - 1, start_epoch - 1)
    atomic_save_checkpoint(
        CKPT_FILE, {"model": model.state_dict(), "config": vars(args), "epoch": last_epoch}
    )

    summary = {
        "data": args.data,
        "n_epochs_run": len(losses),
        "final_loss": losses[-1] if losses else None,
        "initial_loss": losses[0] if losses else None,
        "num_patches": model.num_patches,
        "device": str(device),
    }
    if foxes_batch is not None:
        summary["subsample_n"] = args.subsample_n
        summary["n_train_samples"] = int(foxes_batch[0].shape[0])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-FOXES ViT training / smoke CLI")
    parser.add_argument("--data", choices=["synthetic", "foxes"], default="synthetic")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    # FOXES-Data subsample size (small for the cheap smoke pull) + held-out fraction.
    parser.add_argument("--subsample-n", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    # Architecture knobs (FOXES-faithful defaults).
    parser.add_argument("--in-chans", type=int, default=7)
    parser.add_argument("--img", type=int, default=128)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp", type=int, default=1024)
    parser.add_argument("--local-window", type=int, default=9)
    parser.add_argument("--aux-mask-chans", type=int, default=0)
    args = parser.parse_args()

    try:
        summary = train(args)
        payload = {"config": vars(args), "summary": summary}
    except Exception as exc:  # persist the error so validate-foxes can report it cleanly.
        payload = {"config": vars(args), "summary": {}, "error": repr(exc)}
        atomic_write_json(METRICS_FILE, payload)
        raise

    # Defensive finite-check on the recorded losses before persisting.
    fl = summary.get("final_loss")
    if fl is not None and not math.isfinite(fl):
        payload["error"] = f"non-finite final_loss: {fl}"

    atomic_write_json(METRICS_FILE, payload)
    print(f"wrote {METRICS_FILE}")
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
