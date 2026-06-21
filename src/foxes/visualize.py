"""mini-FOXES XAI visualization -- the repo's first image / heatmap / overlay rendering.

Phase 4 turns the trained checkpoint (Phase 3) into the two figures the HeliolabOps committee
screens for -- spatial attribution and raw attention -- in the repo's established matplotlib
vocabulary (`#1f77b4`, `alpha=0.3`, Agg backend, `dpi=120`, `bbox_inches="tight"`), filling the
documented gap that no image / heatmap / overlay rendering exists anywhere in the repo today.

Two artifacts are produced, mirroring the FOXES paper's own XAI priority:

  1. ``results/foxes_repro/attribution_overlay.png`` -- the PRIMARY artifact. The per-patch
     ``(B, N)`` scalar map (the prediction, decomposed: the patch scalars SUM to the global SXR
     flux) reshaped to the 16x16 patch grid, bilinearly upsampled to 128^2, and overlaid
     (``alpha=0.3``) on a single EUV channel of the held-out input. This *is* the intrinsic
     spatial attribution -- where the model says the X-ray flux comes from.
  2. ``results/foxes_repro/attention_sanity.png`` -- the SECONDARY sanity check. A chosen block /
     head's raw ``(N, N)`` attention matrix reduced to a per-patch "attention received" map
     (column-mean, **no rollout** -- faithful to FOXES, which deliberately avoids rollout /
     Grad-CAM), shown alongside the attribution map for visual comparison.

A small ``viz_meta.json`` sidecar records the quantities ``validate-foxes`` asserts without
re-rendering: that the rendered sample's per-patch attribution **sums to the model's global
prediction** (the visual artifact is faithful to the number), and the two automated
"attribution lands on structure, not noise" proxies -- a non-uniformity (Gini) floor and the
Pearson correlation between per-patch attribution magnitude and per-patch mean EUV brightness.

The held-out sample is pulled the same cheap, streaming, unauthenticated way Phase 2 used:
``load_foxes_subsample`` with a small ``--subsample-n`` (``streaming=True`` + ``.take(n)`` bounds
the pull -- never the full ~1.46 TB). When the HF pull is unavailable, ``--data synthetic`` renders
from a random tensor so the rendering path itself stays testable offline (the validate gate's
structural figure checks then still run; the brightness-correlation proxy is only asserted for the
real-data figures, recorded via ``data_source`` in the sidecar).

Usage:
    python -m src.foxes.visualize                       # real held-out FOXES sample (default)
    python -m src.foxes.visualize --data synthetic       # offline rendering smoke
    python -m src.foxes.visualize --channel 2 --block -1 --head 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / repo convention -- never opens a window.
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .attention import extract_attention
from .model import ViTLocal

OUT_DIR = Path("results/foxes_repro")
ATTRIBUTION_PNG = OUT_DIR / "attribution_overlay.png"
ATTENTION_PNG = OUT_DIR / "attention_sanity.png"
VIZ_META_FILE = OUT_DIR / "viz_meta.json"

# Repo matplotlib vocabulary.
ACCENT = "#1f77b4"
OVERLAY_ALPHA = 0.3
DPI = 120

# Architecture defaults (used only when no checkpoint config is present).
_ARCH_DEFAULTS = dict(
    in_chans=7, img=128, patch=8, embed=256, depth=8, heads=8, mlp=1024, local_window=9
)


def _checkpoint_path() -> Path:
    """Resolve the checkpoint with the SAME logic as run.py / validate.py.

    Prefers ``$STAR_OUTPUT_DIR/foxes_repro/checkpoint.pt`` (external-drive convention) when set
    and the file exists there; otherwise the in-repo ``results/foxes_repro/checkpoint.pt``.
    """
    star = os.environ.get("STAR_OUTPUT_DIR", "")
    if star:
        cand = Path(star) / "foxes_repro" / "checkpoint.pt"
        if cand.exists():
            return cand
    return OUT_DIR / "checkpoint.pt"


def load_trained_model(ckpt_path: Path) -> ViTLocal:
    """Load ``ViTLocal`` with the architecture the checkpoint was trained with + its weights."""
    assert ckpt_path.exists(), f"Missing {ckpt_path} -- run make foxes-train (or pull the ckpt)"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = ViTLocal(
        in_chans=cfg.get("in_chans", _ARCH_DEFAULTS["in_chans"]),
        img=cfg.get("img", _ARCH_DEFAULTS["img"]),
        patch=cfg.get("patch", _ARCH_DEFAULTS["patch"]),
        embed=cfg.get("embed", _ARCH_DEFAULTS["embed"]),
        depth=cfg.get("depth", _ARCH_DEFAULTS["depth"]),
        heads=cfg.get("heads", _ARCH_DEFAULTS["heads"]),
        mlp=cfg.get("mlp", _ARCH_DEFAULTS["mlp"]),
        local_window=cfg.get("local_window", _ARCH_DEFAULTS["local_window"]),
        aux_mask_chans=cfg.get("aux_mask_chans", 0),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_sample(args: argparse.Namespace, model: ViTLocal) -> tuple[torch.Tensor, str, str]:
    """Return a single held-out input ``(1, 7, img, img)`` + a label string + a data-source tag.

    Real path (default): a tiny streaming FOXES-Data pull (unauthenticated, bounded), taking the
    first held-out sample. Synthetic path: a random tensor so the rendering code stays testable
    offline (the brightness-correlation proxy is only meaningful / asserted on real data).
    """
    if args.data == "synthetic":
        g = torch.Generator().manual_seed(args.seed)
        x = torch.randn(1, model.in_chans, model.img, model.img, generator=g)
        return x, "synthetic-random", "synthetic"

    from .data import load_foxes_subsample

    bundle = load_foxes_subsample(
        subsample_n=args.subsample_n,
        img=model.img,
        val_frac=args.val_frac,
        streaming=True,
        seed=args.seed,
    )
    x_val = bundle["x_val"]
    names = bundle["filenames_val"]
    assert x_val.shape[0] >= 1, "no held-out sample materialized; raise --subsample-n"
    idx = min(max(args.sample_index, 0), x_val.shape[0] - 1)
    label = names[idx] if idx < len(names) else f"held-out #{idx}"
    return x_val[idx : idx + 1], str(label), "foxes"


def patch_grid(per_patch: torch.Tensor, model: ViTLocal) -> torch.Tensor:
    """Reshape a ``(N,)`` per-patch vector to the ``(grid_h, grid_w)`` patch grid."""
    return per_patch.reshape(model.grid_h, model.grid_w)


def upsample_to_image(grid: torch.Tensor, img: int) -> torch.Tensor:
    """Bilinearly upsample a ``(gh, gw)`` patch-grid map to ``(img, img)`` image resolution."""
    up = F.interpolate(
        grid[None, None].float(), size=(img, img), mode="bilinear", align_corners=False
    )
    return up[0, 0]


def gini(values: torch.Tensor) -> float:
    """Gini coefficient of a non-negative vector -- a non-uniformity / concentration measure.

    0 == perfectly uniform (every patch contributes equally); ->1 == all mass on one patch. Used
    as the automated "attribution is structured, not flat noise" proxy. Computed on absolute
    magnitudes (attribution can be signed) shifted to be non-negative.
    """
    v = values.flatten().abs().double()
    v = v - v.min() if float(v.min()) < 0 else v
    n = v.numel()
    s = float(v.sum())
    if n == 0 or s == 0:
        return 0.0
    sorted_v, _ = torch.sort(v)
    index = torch.arange(1, n + 1, dtype=torch.double)
    return float((2.0 * (index * sorted_v).sum()) / (n * s) - (n + 1) / n)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two flat vectors (0.0 if either has no variance)."""
    a = a.flatten().double()
    b = b.flatten().double()
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(torch.sqrt((ac**2).sum()) * torch.sqrt((bc**2).sum()))
    return float((ac * bc).sum() / denom) if denom > 0 else 0.0


def render(args: argparse.Namespace) -> dict:
    """Produce both PNGs + the viz_meta.json sidecar; return the metadata dict."""
    ckpt_path = _checkpoint_path()
    model = load_trained_model(ckpt_path)

    x, label, data_source = get_sample(args, model)

    with torch.no_grad():
        global_pred, per_patch = model(x)
    global_pred_v = float(global_pred[0])
    per_patch_v = per_patch[0]  # (N,)
    per_patch_sum = float(per_patch_v.sum())

    # --- spatial maps ---------------------------------------------------------------------
    attr_grid = patch_grid(per_patch_v, model)  # (gh, gw)
    attr_img = upsample_to_image(attr_grid, model.img)  # (img, img)

    # EUV channel used as the overlay base + the brightness reference for the correlation proxy.
    channel = min(max(args.channel, 0), model.in_chans - 1)
    euv = x[0, channel]  # (img, img)

    # Per-patch mean EUV brightness (downsample the channel onto the patch grid) for the
    # attribution<->brightness correlation proxy. AvgPool over patch-sized windows. We compute it
    # for the overlay channel AND for every EUV channel: the SXR flux is driven by the HOT
    # channels (94/131/335 A), so attribution correlates strongly with those and only weakly with
    # the quiet-Sun 171 A -- a single arbitrary channel is therefore a fragile proxy. The robust
    # statistic asserted by the gate is the MEAN per-channel correlation across all EUV channels.
    euv_patch = F.avg_pool2d(euv[None, None], kernel_size=model.patch, stride=model.patch)[0, 0]
    euv_patch_all = F.avg_pool2d(x[0][None], kernel_size=model.patch, stride=model.patch)[0]

    # Raw attention (no rollout): pick a block + head, reduce its (N, N) matrix to a per-patch
    # "attention received" map (mean over query rows -> how much each key patch is attended TO).
    attn = extract_attention(model, x)
    block_keys = sorted(attn.keys())
    block_idx = block_keys[args.block] if attn else None
    attn_grid = None
    if block_idx is not None:
        w = attn[block_idx][0]  # (H, N, N)
        head = min(max(args.head, 0), w.shape[0] - 1)
        per_patch_attn = w[head].mean(dim=0)  # (N,) attention received per key patch
        attn_grid = patch_grid(per_patch_attn, model)

    # --- automated XAI proxies ------------------------------------------------------------
    attr_gini = gini(per_patch_v)
    # Top-decile mass: fraction of total |attribution| in the top 10% of patches (uniform == 0.1).
    mags = per_patch_v.abs().double()
    k = max(1, int(round(0.1 * mags.numel())))
    top_mass = float(mags.sort(descending=True).values[:k].sum() / (mags.sum() + 1e-12))
    brightness_corr = pearson(per_patch_v.abs(), euv_patch)  # overlay channel only (for context)
    per_channel_corr = [
        pearson(per_patch_v.abs(), euv_patch_all[ch]) for ch in range(model.in_chans)
    ]
    # Robust proxy: mean per-channel correlation across all EUV channels (channel-agnostic).
    brightness_corr_mean = sum(per_channel_corr) / len(per_channel_corr)

    # --- figure 1: attribution overlay (PRIMARY) ------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(euv.numpy(), cmap="gray")
    hm = ax.imshow(attr_img.numpy(), cmap="inferno", alpha=OVERLAY_ALPHA)
    ax.set_title(f"Per-patch flux attribution overlay\n{label} (pred {global_pred_v:.3f})")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("per-patch SXR contribution (norm. log-space)")
    fig.savefig(ATTRIBUTION_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # --- figure 2: attention sanity (SECONDARY, alongside attribution) --------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    a0 = axes[0].imshow(attr_grid.numpy(), cmap="inferno")
    axes[0].set_title("Per-patch attribution\n(16x16 grid)", color=ACCENT)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(a0, ax=axes[0], fraction=0.046, pad=0.04)

    if attn_grid is not None:
        a1 = axes[1].imshow(attn_grid.numpy(), cmap="viridis")
        axes[1].set_title(
            f"Raw attention (block {block_idx}, head {args.head})\nattention received, no rollout"
        )
        fig.colorbar(a1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "no attention captured", ha="center", va="center")
        axes[1].set_title("Raw attention (unavailable)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.suptitle("mini-FOXES XAI sanity: attribution vs. raw attention")
    fig.savefig(ATTENTION_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "data_source": data_source,
        "sample_label": label,
        "channel": channel,
        "block_idx": int(block_idx) if block_idx is not None else None,
        "head": int(args.head),
        "global_pred": global_pred_v,
        "per_patch_sum": per_patch_sum,
        "faithfulness_abs_err": abs(per_patch_sum - global_pred_v),
        "attribution_gini": attr_gini,
        "top_decile_mass": top_mass,
        "brightness_pearson_r": brightness_corr,
        "brightness_pearson_r_mean": brightness_corr_mean,
        "brightness_pearson_r_by_channel": per_channel_corr,
        "attribution_overlay_png": str(ATTRIBUTION_PNG),
        "attention_sanity_png": str(ATTENTION_PNG),
        "checkpoint_path": str(ckpt_path),
    }
    tmp = VIZ_META_FILE.with_suffix(VIZ_META_FILE.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"config": vars(args), "viz_meta": meta}, f, indent=2)
    os.replace(tmp, VIZ_META_FILE)

    print(f"wrote {ATTRIBUTION_PNG}")
    print(f"wrote {ATTENTION_PNG}")
    print(f"wrote {VIZ_META_FILE}")
    print("viz_meta:", json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-FOXES XAI visualization")
    parser.add_argument("--data", choices=["synthetic", "foxes"], default="foxes")
    parser.add_argument("--subsample-n", type=int, default=16, help="tiny streaming HF pull")
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--sample-index", type=int, default=0, help="held-out sample to render")
    # Default overlay channel 0 = 94 A, a HOT flaring channel that drives SXR flux (so the
    # attribution<->brightness correspondence is visible in the headline figure); 171 A (ch 2) is
    # quiet-Sun-dominated and correlates only weakly.
    parser.add_argument("--channel", type=int, default=0, help="EUV channel for the overlay base")
    parser.add_argument("--block", type=int, default=-1, help="attention block index (-1 = last)")
    parser.add_argument("--head", type=int, default=0, help="attention head index")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
    # HF `datasets` streaming spins up background HTTP/Arrow threads that can keep the interpreter
    # alive long after the (synchronous) render has finished and all artifacts are flushed -- a
    # plain `return` from main() then leaves `make foxes-figures` hanging. The work is done and
    # outputs are on disk, so exit immediately and deterministically.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
