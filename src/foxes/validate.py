"""mini-FOXES correctness gate (the `validate-foxes` Makefile target).

In the repo's inline-Python `assert`-gate tradition (validate-baseline / validate-eval), but
factored into a module because the FOXES faithfulness invariants are too many to read as a
single shell one-liner. Run via `python -m src.foxes.validate`.

Phase 1 assertions:
  * results/foxes_repro/metrics.json exists, is finite, and has no `error` key;
  * forward output shapes are (B,) and (B, N) with N = (img/patch)^2 = 256 at 128/8;
  * FAITHFULNESS invariant: |per_patch.sum(1) - global| < 1e-4;
  * extracted attention is non-None with shape (B, H, N, N);
  * the inverted mask blocks the 9x9 neighborhood (local entries blocked, distant allowed);
  * the saved checkpoint reloads and reproduces the forward pass.
  * FOXES STRUCTURAL invariants: no CLS token (token count N, no cls_token/class_token attr);
    per-patch head is Linear(embed, 1); every encoder block is an InvertedAttentionBlock.

Phase 2 assertions (only when results/foxes_repro/data_smoke.json exists):
  * real input tensor shape (n, 7, 128, 128), dtype float32;
  * value range within [-1, 1] +/- eps (the authors' ITI normalization);
  * normalized log-space labels are finite;
  * STREAMING-BOUNDED pull: is_streaming is True AND exactly subsample_n records were
    materialized (so no full ~1.46 TB download can have occurred).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from .attention import extract_attention
from .model import InvertedAttentionBlock, ViTLocal

OUT_DIR = Path("results/foxes_repro")
METRICS_FILE = OUT_DIR / "metrics.json"
CKPT_FILE = OUT_DIR / "checkpoint.pt"
DATA_SMOKE_FILE = OUT_DIR / "data_smoke.json"


def check_metrics() -> dict:
    assert METRICS_FILE.exists(), f"Missing {METRICS_FILE} -- run make foxes-smoke first"
    d = json.load(open(METRICS_FILE))
    assert "error" not in d, f"metrics.json has error: {d.get('error')}"
    summary = d.get("summary", {})
    finite_vals = [summary[k] for k in ("final_loss", "initial_loss") if summary.get(k) is not None]
    assert finite_vals, "no loss recorded in summary"
    assert all(math.isfinite(v) for v in finite_vals), f"non-finite loss: {finite_vals}"
    print(f"  metrics OK: {summary}")
    return d


def check_data_smoke() -> None:
    """Phase-2 gate: assert the real FOXES-Data smoke artifact (only when it exists).

    Skips silently if data_smoke.json is absent (Phase-1-only runs), so synthetic-only
    validation still passes. When present, asserts tensor shape/dtype/value-range, finite
    labels, and the streaming-bounded invariant (the automated stand-in for the manual
    "did it pull the full 1.46 TB?" check).
    """
    if not DATA_SMOKE_FILE.exists():
        print("  data smoke: (skipped -- no data_smoke.json; synthetic-only run)")
        return
    d = json.load(open(DATA_SMOKE_FILE))
    smoke = d.get("data_smoke", {})

    shape = smoke.get("tensor_shape")
    assert shape and len(shape) == 4, f"bad tensor_shape {shape}"
    n, c, h, w = shape
    assert c == 7, f"expected 7 EUV channels, got {c}"
    assert h == 128 and w == 128, f"expected 128x128, got {h}x{w}"
    assert "float32" in smoke.get("dtype", ""), f"dtype not float32: {smoke.get('dtype')}"

    eps = 1e-3
    vmin, vmax = smoke["value_min"], smoke["value_max"]
    assert vmin >= -1.0 - eps and vmax <= 1.0 + eps, (
        f"value range [{vmin}, {vmax}] outside [-1, 1]+/-eps"
    )

    lmin, lmax = smoke["label_min"], smoke["label_max"]
    assert math.isfinite(lmin) and math.isfinite(lmax), f"non-finite labels [{lmin}, {lmax}]"
    assert smoke.get("labels_finite") is True, "labels_finite flag is not True"

    # Streaming-bounded download invariant.
    assert smoke.get("is_streaming") is True, "loader was not constructed with streaming=True"
    sub_n = smoke.get("subsample_n")
    n_mat = smoke.get("n_materialized")
    assert n_mat == sub_n, f"materialized {n_mat} != subsample_n {sub_n} (unbounded pull?)"
    print(
        f"  data smoke OK: shape {shape} float32, value [{vmin:.3f}, {vmax:.3f}], "
        f"labels finite, streaming pull bounded to {n_mat} records"
    )


def check_structural(model: ViTLocal) -> None:
    # No CLS / class token.
    assert not hasattr(model, "cls_token"), "model has a cls_token (FOXES uses none)"
    assert not hasattr(model, "class_token"), "model has a class_token (FOXES uses none)"
    expected_n = (model.img // model.patch) ** 2
    assert model.num_patches == expected_n, f"num_patches {model.num_patches} != {expected_n}"
    assert model.pos_embed.shape[1] == expected_n, "pos_embed has a CLS slot (N+1)"

    # Per-patch head is Linear(embed, 1).
    head_linears = [m for m in model.mlp_head.modules() if isinstance(m, nn.Linear)]
    assert len(head_linears) == 1, f"expected one Linear in head, got {len(head_linears)}"
    head = head_linears[0]
    assert head.in_features == model.embed and head.out_features == 1, (
        f"head must be Linear({model.embed}, 1), "
        f"got Linear({head.in_features}, {head.out_features})"
    )

    # Every encoder block is an InvertedAttentionBlock (no standard-attention layer).
    assert len(model.blocks) == model.depth, "block count != depth"
    for i, blk in enumerate(model.blocks):
        assert isinstance(blk, InvertedAttentionBlock), f"block {i} is not InvertedAttentionBlock"
    print(f"  structural OK: no CLS, head=Linear({model.embed},1), {model.depth} inverted blocks")


def check_mask(model: ViTLocal) -> None:
    mask = model.attn_mask  # (N, N) bool, True == blocked
    n = model.num_patches
    assert mask.shape == (n, n), f"mask shape {tuple(mask.shape)} != ({n},{n})"
    assert mask.dtype == torch.bool, f"mask dtype {mask.dtype} != bool"
    # Diagonal (self) and immediate neighbors are within the 9x9 neighborhood -> blocked.
    assert bool(mask[0, 0]), "patch-0 self must be blocked (within 9x9 neighborhood)"
    assert bool(mask[0, 1]), "patch-0 -> adjacent patch must be blocked (local)"
    # A distant patch (opposite corner) must be allowed.
    assert not bool(mask[0, n - 1]), "distant patch (corner) must be allowed (non-local)"
    # Each row must allow at least one distant patch (else attention has no valid targets).
    allowed_per_row = (~mask).sum(dim=1)
    min_allowed = int(allowed_per_row.min())
    assert min_allowed > 0, "some patch has no allowed (distant) targets"
    print(f"  mask OK: 9x9 neighborhood blocked, distant allowed (min allowed/row={min_allowed})")


def check_forward_and_faithfulness(model: ViTLocal) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    b, n = 2, model.num_patches
    torch.manual_seed(0)
    x = torch.randn(b, model.in_chans, model.img, model.img)
    with torch.no_grad():
        global_pred, per_patch = model(x)
    assert tuple(global_pred.shape) == (b,), f"global shape {tuple(global_pred.shape)} != ({b},)"
    assert tuple(per_patch.shape) == (b, n), (
        f"per_patch shape {tuple(per_patch.shape)} != ({b},{n})"
    )
    diff = (per_patch.sum(1) - global_pred).abs().max().item()
    assert diff < 1e-4, f"faithfulness invariant violated: max |sum-global| = {diff}"
    print(f"  forward OK: shapes ({b},) and ({b},{n}); faithfulness max|sum-global|={diff:.2e}")
    return x, global_pred


def check_attention(model: ViTLocal, x: torch.Tensor) -> None:
    attn = extract_attention(model, x)
    assert attn, "no attention captured"
    n = model.num_patches
    for idx, w in attn.items():
        assert w is not None, f"block {idx} attention is None"
        assert tuple(w.shape) == (x.shape[0], model.heads, n, n), (
            f"block {idx} attn shape {tuple(w.shape)} != ({x.shape[0]},{model.heads},{n},{n})"
        )
    assert len(attn) == model.depth, f"captured {len(attn)} blocks, expected {model.depth}"
    print(
        f"  attention OK: {len(attn)} blocks, each (B,H,N,N)=({x.shape[0]},{model.heads},{n},{n})"
    )


def check_checkpoint_reload(x: torch.Tensor, ref_global: torch.Tensor) -> None:
    assert CKPT_FILE.exists(), f"Missing {CKPT_FILE} -- run make foxes-smoke first"
    ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ViTLocal(
        in_chans=cfg["in_chans"],
        img=cfg["img"],
        patch=cfg["patch"],
        embed=cfg["embed"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        mlp=cfg["mlp"],
        local_window=cfg["local_window"],
        aux_mask_chans=cfg["aux_mask_chans"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        reloaded_global, _ = model(x)
    diff = (reloaded_global - ref_global).abs().max().item()
    assert diff < 1e-4, f"checkpoint reload mismatch: max diff {diff}"
    print(f"  checkpoint OK: reloaded model reproduces forward (max diff {diff:.2e})")


def main() -> None:
    print("validate-foxes:")
    check_metrics()
    check_data_smoke()

    # Build a fresh model with the SAME architecture the checkpoint used, so the structural,
    # mask, forward, attention, and reload checks all share one configuration.
    ckpt = (
        torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
        if CKPT_FILE.exists()
        else None
    )
    cfg = ckpt["config"] if ckpt else {}
    model = ViTLocal(
        in_chans=cfg.get("in_chans", 7),
        img=cfg.get("img", 128),
        patch=cfg.get("patch", 8),
        embed=cfg.get("embed", 256),
        depth=cfg.get("depth", 8),
        heads=cfg.get("heads", 8),
        mlp=cfg.get("mlp", 1024),
        local_window=cfg.get("local_window", 9),
        aux_mask_chans=cfg.get("aux_mask_chans", 0),
    )
    if ckpt:
        model.load_state_dict(ckpt["model"])

    check_structural(model)
    check_mask(model)
    x, ref_global = check_forward_and_faithfulness(model)
    check_attention(model, x)
    check_checkpoint_reload(x, ref_global)
    print("validate-foxes OK")


if __name__ == "__main__":
    main()
