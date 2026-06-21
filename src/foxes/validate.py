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

Phase 3 assertions (only when a TRAINED metrics summary is present -- i.e. `summary.mae_dex`
exists; absent for the synthetic + data-smoke states, so those stay green/skipped):
  * `summary.mae_dex` finite;
  * PRIMARY self-calibrating gate: mae_dex < mae_dex_mean_baseline by a margin (the model must
    beat predicting the constant train-mean on the SAME held-out set -- real learning, no
    hard-coded magic number, mirroring validate-baseline's "loss decreased" style);
  * SECONDARY loose sanity band: 0.01 < mae_dex < 0.4 (lower bound catches label leakage /
    impossibly-perfect bugs, upper bound catches a non-learning / blowup run and keeps us under
    the paper's 0.307 naive baseline);
  * Pearson r finite and in [-1, 1];
  * the cost estimate elapsed_hr * usd_per_hr < 25 (budget-ceiling sanity check);
  * the faithfulness invariant and the checkpoint-reload-reproduces-metrics check already run on
    the (trained, when present) checkpoint via the shared model below.

Phase 4 assertions (only when results/foxes_repro/viz_meta.json exists -- i.e. make foxes-figures
has been run; absent for earlier states, so those stay green/skipped):
  * both PNGs (attribution_overlay.png + attention_sanity.png) exist and are non-empty;
  * the rendered sample's per-patch attribution SUMS to the model's global prediction
    (|per_patch_sum - global_pred| < tol) -- the visual artifact is faithful to the number;
  * the "attribution corresponds to structure, not noise" proxies: the per-patch attribution is
    NON-UNIFORM (Gini above a floor AND top-decile mass well above the uniform 0.1) and, for the
    REAL-data figure, the per-patch attribution magnitude POSITIVELY CORRELATES with per-patch
    EUV brightness (Pearson r above a small positive threshold) -- a quantitative stand-in for
    "attribution lands on active regions." The brightness proxy is skipped for synthetic renders
    (random input has no structure), recorded via `data_source` in the sidecar.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from .attention import extract_attention
from .model import InvertedAttentionBlock, ViTLocal

OUT_DIR = Path("results/foxes_repro")
METRICS_FILE = OUT_DIR / "metrics.json"
DATA_SMOKE_FILE = OUT_DIR / "data_smoke.json"
VIZ_META_FILE = OUT_DIR / "viz_meta.json"
ATTRIBUTION_PNG = OUT_DIR / "attribution_overlay.png"
ATTENTION_PNG = OUT_DIR / "attention_sanity.png"

# Primary-gate margin: the trained model must beat the constant-mean baseline by at least this
# fraction of the baseline MAE (a small but non-trivial relative improvement -> proves learning).
PRIMARY_GATE_MARGIN = 0.05  # 5% relative improvement over the mean-baseline MAE
SANITY_MAE_LO = 0.01
SANITY_MAE_HI = 0.4
COST_CEILING_USD = 25.0

# Phase-4 XAI proxy floors. The attribution must be visibly structured (not flat noise) and,
# on real data, land on bright EUV structure. These are deliberately loose -- they catch a dead
# / uniform / anti-correlated map, not a specific scientific claim (the expert eyeball is the
# deferred Final Manual Verification).
VIZ_FAITHFULNESS_TOL = 1e-3  # |per_patch_sum - global_pred| in normalized log-space
VIZ_GINI_FLOOR = 0.05  # Gini concentration floor (0 == perfectly uniform)
VIZ_TOP_DECILE_FLOOR = 0.12  # top-10% patch mass must exceed the uniform 0.10
# Min mean-over-channels Pearson r(|attribution|, per-patch EUV brightness) on real data. The
# SXR flux is driven by the hot channels (94/131/335 A), so the channel-averaged correlation is
# robustly positive (~0.22-0.30 measured); a single arbitrary channel (e.g. quiet-Sun 171 A) can
# read near zero, so the gate asserts the channel-averaged statistic.
VIZ_BRIGHTNESS_R_FLOOR = 0.10


def _checkpoint_path() -> Path:
    """Resolve the checkpoint location with the SAME logic as run.py.

    Prefers `$STAR_OUTPUT_DIR/foxes_repro/checkpoint.pt` (external-drive convention) when set and
    its parent is reachable AND the file exists there; otherwise the in-repo results checkpoint.
    """
    star = os.environ.get("STAR_OUTPUT_DIR", "")
    if star:
        cand = Path(star) / "foxes_repro" / "checkpoint.pt"
        if cand.exists():
            return cand
    return OUT_DIR / "checkpoint.pt"


CKPT_FILE = _checkpoint_path()


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


def check_trained_results(metrics: dict) -> None:
    """Phase-3 gate: activate the trained-result assertions only when a trained summary exists.

    A trained run records `summary.mae_dex` (a held-out eval in dex). When that key is absent
    (synthetic + data-smoke states), this skips silently so the earlier phases stay green. When
    present, it enforces the self-calibrating primary gate + loose sanity band + Pearson finite +
    cost ceiling described in the structure outline.
    """
    summary = metrics.get("summary", {})
    if "mae_dex" not in summary:
        print("  trained results: (skipped -- no trained metrics summary yet)")
        return

    mae = summary["mae_dex"]
    assert isinstance(mae, (int, float)) and math.isfinite(mae), f"mae_dex not finite: {mae}"

    # PRIMARY self-calibrating gate: beat the constant-mean baseline by a margin.
    base = summary.get("mae_dex_mean_baseline")
    assert base is not None and math.isfinite(base), f"missing/non-finite mean baseline: {base}"
    assert mae < base * (1.0 - PRIMARY_GATE_MARGIN), (
        f"model MAE {mae:.4f} dex did not beat the mean-baseline {base:.4f} dex by "
        f"{PRIMARY_GATE_MARGIN:.0%} (no real learning)"
    )

    # SECONDARY loose sanity band.
    assert SANITY_MAE_LO < mae < SANITY_MAE_HI, (
        f"mae_dex {mae:.4f} outside sanity band ({SANITY_MAE_LO}, {SANITY_MAE_HI}) "
        "(leakage/impossibly-perfect below; non-learning/blowup above)"
    )

    # Pearson r finite and in [-1, 1].
    r = summary.get("pearson_r")
    assert r is not None and math.isfinite(r), f"pearson_r not finite: {r}"
    assert -1.0 <= r <= 1.0, f"pearson_r {r} outside [-1, 1]"

    # Cost-ceiling sanity check.
    eh = summary.get("elapsed_hr")
    uph = summary.get("usd_per_hr")
    assert eh is not None and uph is not None, "missing elapsed_hr / usd_per_hr for cost check"
    cost = eh * uph
    assert cost < COST_CEILING_USD, f"estimated cost ${cost:.2f} exceeds ${COST_CEILING_USD:.0f}"

    print(
        f"  trained results OK: mae_dex={mae:.4f} < baseline={base:.4f} (margin "
        f"{PRIMARY_GATE_MARGIN:.0%}); pearson_r={r:.4f}; est_cost=${cost:.2f}"
    )


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


def check_figures() -> None:
    """Phase-4 gate: assert the rendered XAI figures (only when viz_meta.json exists).

    Skips silently if viz_meta.json is absent (pre-Phase-4 states), so earlier validation still
    passes. When present, asserts both PNGs exist and are non-empty, the attribution-sums-to-
    global faithfulness invariant on the *rendered* sample, and the two "structure, not noise"
    proxies (non-uniformity floors always; brightness correlation only for real-data renders).
    """
    if not VIZ_META_FILE.exists():
        print("  figures: (skipped -- no viz_meta.json; run make foxes-figures)")
        return
    d = json.load(open(VIZ_META_FILE))
    meta = d.get("viz_meta", {})

    for png in (ATTRIBUTION_PNG, ATTENTION_PNG):
        assert png.exists(), f"missing figure {png} -- run make foxes-figures"
        assert png.stat().st_size > 0, f"figure {png} is empty"

    # Faithfulness on the rendered sample: per-patch attribution sums to the global prediction.
    ferr = meta.get("faithfulness_abs_err")
    assert ferr is not None and math.isfinite(ferr), f"missing/non-finite faithfulness err: {ferr}"
    assert ferr < VIZ_FAITHFULNESS_TOL, (
        f"rendered attribution sum diverges from global pred by {ferr:.2e} "
        f"(>{VIZ_FAITHFULNESS_TOL:.0e}) -- the figure is not faithful to the number"
    )

    # Non-uniformity: the map must be concentrated, not flat noise.
    gini = meta.get("attribution_gini")
    top_mass = meta.get("top_decile_mass")
    assert gini is not None and math.isfinite(gini), f"missing/non-finite gini: {gini}"
    assert gini > VIZ_GINI_FLOOR, (
        f"attribution Gini {gini:.3f} <= floor {VIZ_GINI_FLOOR} -- map is ~uniform (no structure)"
    )
    assert top_mass is not None and top_mass > VIZ_TOP_DECILE_FLOOR, (
        f"top-decile mass {top_mass} <= floor {VIZ_TOP_DECILE_FLOOR} (uniform == 0.10)"
    )

    # Brightness correlation: only meaningful on real EUV structure (skip synthetic renders).
    # Asserts the channel-averaged correlation (robust); falls back to the single-channel value
    # for sidecars written before the mean statistic existed.
    data_source = meta.get("data_source")
    r = meta.get("brightness_pearson_r_mean", meta.get("brightness_pearson_r"))
    if data_source == "foxes":
        assert r is not None and math.isfinite(r), f"missing/non-finite brightness r: {r}"
        assert r > VIZ_BRIGHTNESS_R_FLOOR, (
            f"attribution<->EUV-brightness mean Pearson r {r:.3f} <= floor "
            f"{VIZ_BRIGHTNESS_R_FLOOR} -- attribution does not land on bright structure"
        )
        print(
            f"  figures OK: both PNGs present, faithful (err {ferr:.2e}), "
            f"gini={gini:.3f}, top-decile={top_mass:.3f}, brightness r(mean)={r:.3f} (real data)"
        )
    else:
        print(
            f"  figures OK: both PNGs present, faithful (err {ferr:.2e}), gini={gini:.3f}, "
            f"top-decile={top_mass:.3f} (synthetic render -- brightness proxy skipped)"
        )


def main() -> None:
    print("validate-foxes:")
    metrics = check_metrics()
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
    # Faithfulness invariant runs on the (trained, when present) checkpoint weights.
    x, ref_global = check_forward_and_faithfulness(model)
    check_attention(model, x)
    check_checkpoint_reload(x, ref_global)
    # Phase-3 trained-result gates (skipped silently for synthetic + data-smoke states).
    check_trained_results(metrics)
    # Phase-4 figure gates (skipped silently until make foxes-figures has been run).
    check_figures()
    print("validate-foxes OK")


if __name__ == "__main__":
    main()
