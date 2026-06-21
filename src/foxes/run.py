"""mini-FOXES training / smoke CLI (Makefile-driven).

Phase 1 wired `--data synthetic`: random `(B, 7, 128, 128)` inputs + random log-SXR targets,
runnable on CPU with no data download and no GPU -- the documented Option-B fallback.

Phase 2 wires `--data foxes`: a *tiny* streaming subsample of the FOXES authors' own published
dataset (`griffingoodwin04/FOXES-Data`) flows through the Phase-1 network + loss on CPU. The
real records are resized 512^2 -> 128^2 and their GOES SXR flux is converted to a normalized
log-space label (see `data.py`). A small `data_smoke.json` artifact records tensor shape /
dtype / value-range / label stats so the `validate-foxes` gate can assert the real path works
without re-downloading.

Phase 3 turns this into a *real* training loop, run on cloud CUDA (A6000/4090 via Vast.ai;
avoids the M3-MPS masked-attention NaN bug):

  * Huber loss (delta=0.3) in normalized log-space (the FOXES loss);
  * cosine LR schedule 1e-4 -> ~1e-5 over `--epochs` (20-50);
  * `--checkpoint-every` + `--resume` (rebuild done / last-epoch state from a checkpoint);
  * an eval pass over the held-out subset computing MAE / RMSE / Pearson r in **dex**
    (log10 W/m^2) + mean bias, plus `mae_dex_mean_baseline` -- the MAE of a trivial constant
    predictor (the training-set mean log-SXR) on the SAME held-out set. The primary
    `validate-foxes` gate is self-calibrating: the model must beat that baseline (real learning,
    no hard-coded magic number);
  * elapsed wall-clock + the instance `$/hr` recorded into `summary` for an automated
    cost-ceiling check (`elapsed_hr * usd_per_hr < 25`).

This pins the public signatures (the "C header" of the showcase) and produces a
provenance-shaped `metrics.json` plus an atomic checkpoint, so the `validate-foxes`
faithfulness gate can run before any cloud money is spent on the full training run.

Provenance follows the repo convention (eval_vision.py / train_lstm.py):
  * every output JSON embeds `"config": vars(args) | {...}`,
  * writes are atomic (temp file + os.replace),
  * the training loop supports `--resume` and `--checkpoint-every` (default 250),
  * the trained checkpoint lands under `STAR_OUTPUT_DIR` (external-drive convention) while the
    small metrics.json stays in the repo tree.

Usage:
    python -m src.foxes.run --data synthetic --epochs 1                       # CPU smoke (no data)
    python -m src.foxes.run --data foxes --subsample-n 8 --epochs 1           # tiny real data pull
    python -m src.foxes.run --data foxes --subsample-n 1500 --epochs 40 \
        --device cuda --usd-per-hr 0.40                       # the Phase-3 cloud run
    python -m src.foxes.run --data foxes --subsample-n 1500 --epochs 40 --resume   # resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

from .model import ViTLocal

OUT_DIR = Path("results/foxes_repro")
METRICS_FILE = OUT_DIR / "metrics.json"
DATA_SMOKE_FILE = OUT_DIR / "data_smoke.json"

# The trained checkpoint lands under STAR_OUTPUT_DIR (the external-drive convention, the same
# env var the LSTM baseline uses) so the multi-MB weights never bloat the repo / internal disk;
# it falls back to the in-repo results dir when the drive is not mounted (e.g. CPU smoke runs).
STAR_OUTPUT_DIR = os.environ.get("STAR_OUTPUT_DIR", "")


def checkpoint_path() -> Path:
    """Where the trained checkpoint is written/read.

    Prefers `$STAR_OUTPUT_DIR/foxes_repro/checkpoint.pt` (external drive) when STAR_OUTPUT_DIR
    is set AND its parent is reachable; otherwise the in-repo `results/foxes_repro/checkpoint.pt`
    (CPU smoke / CI). validate-foxes resolves the checkpoint with the identical logic.
    """
    if STAR_OUTPUT_DIR:
        root = Path(STAR_OUTPUT_DIR)
        if root.parent.exists():
            return root / "foxes_repro" / "checkpoint.pt"
    return OUT_DIR / "checkpoint.pt"


CKPT_FILE = checkpoint_path()


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
    """Pull the FOXES-Data subsample and write a `data_smoke.json` provenance artifact.

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


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch: int, generator: torch.Generator):
    """Yield shuffled `(xb, yb)` minibatches for one epoch over a held-in tensor set."""
    n = x.shape[0]
    perm = torch.randperm(n, generator=generator)
    for start in range(0, n, batch):
        idx = perm[start : start + batch]
        yield x[idx], y[idx]


@torch.no_grad()
def evaluate_dex(
    model: ViTLocal,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    batch: int,
) -> dict:
    """Held-out eval in **dex** (log10 W/m^2 -- the unit FOXES reports MAE/RMSE in).

    Predictions and labels are both in normalized log-space; `denormalize_log_sxr` maps them
    back to dex so MAE / RMSE / mean bias are directly comparable to the paper (MAE 0.051 dex,
    naive baseline 0.307). Pearson r is scale-invariant so it is identical in either space.
    """
    from .data import denormalize_log_sxr

    model.eval()
    preds_norm: list[float] = []
    for start in range(0, x_val.shape[0], batch):
        xb = x_val[start : start + batch].to(device)
        gp, _ = model(xb)
        preds_norm.extend(gp.detach().cpu().tolist())

    half = denormalize_log_sxr(1.0) - denormalize_log_sxr(0.0)  # norm->dex scale factor
    pred_dex = torch.tensor([denormalize_log_sxr(p) for p in preds_norm], dtype=torch.float64)
    true_dex = torch.tensor(
        [denormalize_log_sxr(float(t)) for t in y_val.tolist()], dtype=torch.float64
    )

    err = pred_dex - true_dex
    mae = float(err.abs().mean())
    rmse = float(torch.sqrt((err**2).mean()))
    bias = float(err.mean())

    # Pearson r in dex (finite-guarded: needs variance in both series).
    pred_c = pred_dex - pred_dex.mean()
    true_c = true_dex - true_dex.mean()
    denom = float(torch.sqrt((pred_c**2).sum()) * torch.sqrt((true_c**2).sum()))
    pearson_r = float((pred_c * true_c).sum() / denom) if denom > 0 else 0.0

    # Self-calibrating baseline: a constant predictor = the TRAIN mean log-SXR, scored on the
    # SAME held-out set, in dex. The model must beat this to prove it learned anything.
    return {
        "n_val": int(y_val.shape[0]),
        "mae_dex": mae,
        "rmse_dex": rmse,
        "mean_bias_dex": bias,
        "pearson_r": pearson_r,
        "_norm_to_dex_half": float(half),
    }


def mean_baseline_mae_dex(y_train: torch.Tensor, y_val: torch.Tensor) -> float:
    """MAE in dex of the trivial constant predictor (train-mean log-SXR) on the held-out set."""
    from .data import denormalize_log_sxr

    const_norm = float(y_train.mean())
    const_dex = denormalize_log_sxr(const_norm)
    true_dex = torch.tensor(
        [denormalize_log_sxr(float(t)) for t in y_val.tolist()], dtype=torch.float64
    )
    return float((true_dex - const_dex).abs().mean())


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    t0 = time.time()

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Cosine LR 1e-4 -> ~1e-5 over the full epoch budget (the FOXES schedule).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr_min
    )
    loss_fn = nn.HuberLoss(delta=0.3)

    # Data: the real FOXES-Data subsample (train/held-out split) vs. synthetic random tensors.
    bundle: dict | None = None
    x_train = y_train = x_val = y_val = None
    if args.data == "foxes":
        bundle = load_foxes_data(args)
        x_train, y_train = bundle["x_train"], bundle["y_train"]
        x_val, y_val = bundle["x_val"], bundle["y_val"]

    start_epoch = 0
    if args.resume and CKPT_FILE.exists():
        ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        ckpt_epoch = ckpt.get("epoch")
        start_epoch = int(ckpt_epoch or 0) + 1
        print(f"[resume] loaded epoch {ckpt_epoch}, continuing from {start_epoch}")

    def save_ckpt(epoch: int) -> None:
        atomic_save_checkpoint(
            CKPT_FILE,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": vars(args),
                "epoch": epoch,
            },
        )

    losses: list[float] = []
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses: list[float] = []
        if args.data == "foxes":
            batches = iterate_minibatches(x_train, y_train, args.batch, gen)
        else:
            # Synthetic: a fresh random batch each step (Option-B fallback, no real data).
            batches = (synthetic_batch(args, gen) for _ in range(args.synthetic_steps))
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            global_pred, _per_patch = model(xb)
            loss = loss_fn(global_pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(mean_loss)
        last_epoch = epoch
        print(f"[epoch {epoch}] loss={mean_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

        if args.checkpoint_every and ((epoch + 1) % args.checkpoint_every == 0):
            save_ckpt(epoch)

    # Always save a final checkpoint so --resume and the reload assertion have something.
    save_ckpt(max(last_epoch, start_epoch - 1))

    elapsed_s = time.time() - t0
    summary = {
        "data": args.data,
        "n_epochs_run": len(losses),
        "epochs_total": args.epochs,
        "final_loss": losses[-1] if losses else None,
        "initial_loss": losses[0] if losses else None,
        "num_patches": model.num_patches,
        "device": str(device),
        "elapsed_s": elapsed_s,
        "elapsed_hr": elapsed_s / 3600.0,
        "usd_per_hr": args.usd_per_hr,
        "est_cost_usd": (elapsed_s / 3600.0) * args.usd_per_hr,
    }

    # Trained-result block: only the real FOXES path has a held-out set to score.
    if args.data == "foxes":
        assert x_val is not None and y_val is not None
        if int(y_val.shape[0]) >= 1:
            metrics = evaluate_dex(model, x_val, y_val, device, args.batch)
            summary.update({k: v for k, v in metrics.items() if not k.startswith("_")})
            summary["mae_dex_mean_baseline"] = mean_baseline_mae_dex(y_train, y_val)
        summary["subsample_n"] = args.subsample_n
        summary["n_train_samples"] = int(x_train.shape[0])
        summary["checkpoint_path"] = str(CKPT_FILE)

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
    parser.add_argument("--lr-min", type=float, default=1e-5, help="cosine schedule floor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    # Synthetic path: number of random minibatch steps per epoch (Option-B fallback only).
    parser.add_argument("--synthetic-steps", type=int, default=1)
    # Cost accounting for the automated budget-ceiling check (Vast.ai A6000 ~0.40, 4090 ~0.37).
    parser.add_argument("--usd-per-hr", type=float, default=0.40)
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
