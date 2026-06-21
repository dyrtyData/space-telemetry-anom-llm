"""Real FOXES-Data path: streaming HF subsample -> 7x128^2 tensors + normalized log-SXR labels.

Phase 2 replaces the Phase-1 synthetic random tensors with the FOXES authors' own published,
*pre-paired* dataset (`griffingoodwin04/FOXES-Data`; Goodwin et al. 2026, ApJ,
DOI 10.3847/1538-4357/ae5e4a). Each record is:

  * `aia_stack`  -- (7, 512, 512) float32 SDO/AIA EUV stack (94/131/171/193/211/304/335 A),
                    already ITI-preprocessed and normalized to [-1, 1] by the authors;
  * `sxr_value`  -- the contemporaneous GOES 1-8 A (XRSB) soft-X-ray flux in W/m^2 (a scalar,
                    delivered as a length-1 list by the HF schema);
  * `filename`   -- an ISO-timestamp string identifying the snapshot.

The two operations this module owns (everything else is already preprocessed upstream):

  1. Resize 512^2 -> 128^2 with a single `F.interpolate` (bilinear) so a tiny ViT trains
     cheaply on CPU/consumer-GPU -- the FOXES paper uses 512^2; we keep a faithful *miniature*.
  2. Convert the raw W/m^2 SXR flux to a normalized log-space target (the regression label),
     matching the loss space the training run (Phase 3) optimizes in.

Streaming is mandatory: `load_dataset(..., streaming=True).take(n)` materializes exactly `n`
records, so a tiny smoke pull never triggers the full ~1.46 TB download. The loader is an
`IterableDataset` and yields at most `--subsample-n` records, both of which the `validate-foxes`
gate asserts (automating the former manual "did it pull the whole dataset?" check).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

DATASET_REPO = "griffingoodwin04/FOXES-Data"
EUV_CHANNELS = (94, 131, 171, 193, 211, 304, 335)  # SDO/AIA EUV wavelengths (angstrom)

# Normalized log-space target band. GOES XRSB flare classes span ~1e-8 (A) to >1e-3 (X) W/m^2,
# i.e. log10(flux) in roughly [-8, -3]. We map that physical band affinely to ~[-1, 1] so the
# regression target sits in a small, well-conditioned range for the Huber loss (Phase 3 reuses
# these constants, so the smoke labels and the training labels live in the SAME space).
LOG_SXR_MIN = -8.0  # log10(W/m^2) lower anchor (sub-A-class background)
LOG_SXR_MAX = -3.0  # log10(W/m^2) upper anchor (above X10)
_LOG_MID = (LOG_SXR_MAX + LOG_SXR_MIN) / 2.0
_LOG_HALF = (LOG_SXR_MAX - LOG_SXR_MIN) / 2.0
_FLUX_FLOOR = 1e-9  # guards log10 against zero / negative flux values


def normalize_log_sxr(flux_wm2: float) -> float:
    """Raw GOES XRSB flux (W/m^2) -> normalized log-space target in ~[-1, 1].

    `(log10(flux) - mid) / half` over the [LOG_SXR_MIN, LOG_SXR_MAX] band. Values are NOT
    clamped (a rare super-X event may exceed 1.0), but the floor keeps log10 finite.
    """
    flux = max(float(flux_wm2), _FLUX_FLOOR)
    return (math.log10(flux) - _LOG_MID) / _LOG_HALF


def denormalize_log_sxr(norm: float) -> float:
    """Inverse of `normalize_log_sxr`: normalized target -> log10(flux) in dex.

    Returns the prediction in dex (log10 W/m^2), the unit the FOXES MAE/RMSE are reported in.
    """
    return norm * _LOG_HALF + _LOG_MID


def _scalar_sxr(raw) -> float:
    """Coerce the HF `sxr_value` (a length-1 list per the schema) to a python float."""
    if isinstance(raw, (list, tuple)):
        assert len(raw) >= 1, "empty sxr_value"
        return float(raw[0])
    return float(raw)


def resize_stack(aia_stack: Tensor, img: int) -> Tensor:
    """Resize a (7, 512, 512) EUV stack to (7, img, img) with one bilinear `F.interpolate`."""
    if aia_stack.dim() == 3:
        aia_stack = aia_stack.unsqueeze(0)  # (1, 7, H, W)
    resized = F.interpolate(aia_stack, size=(img, img), mode="bilinear", align_corners=False)
    return resized.squeeze(0)  # (7, img, img)


def load_foxes_subsample(
    subsample_n: int,
    img: int = 128,
    val_frac: float = 0.25,
    streaming: bool = True,
    split: str = "train",
):
    """Stream `subsample_n` FOXES-Data records and return train/held-out tensor batches.

    Returns a dict with:
      * `x_train`, `y_train`, `x_val`, `y_val` -- float32 tensors, inputs (n, 7, img, img),
        labels (n,) in normalized log-space;
      * `filenames_train`, `filenames_val` -- the ISO-timestamp ids;
      * `is_streaming` -- True (the loader was built with `streaming=True`);
      * `n_materialized` -- how many records were actually pulled (must equal `subsample_n`).

    The held-out split is a deterministic tail slice (the last `round(n * val_frac)` records),
    so it is reproducible without shuffling a streamed iterator.
    """
    from datasets import load_dataset

    assert subsample_n >= 1, "subsample_n must be >= 1"
    ds = load_dataset(DATASET_REPO, streaming=streaming, split=split)
    # `streaming=True` yields an IterableDataset; `.take(n)` bounds the pull to n records so the
    # full ~1.46 TB dataset can never be downloaded by accident.
    bounded = ds.take(subsample_n) if streaming else ds.select(range(subsample_n))

    xs: list[Tensor] = []
    ys: list[float] = []
    names: list[str] = []
    for rec in bounded:
        stack = torch.as_tensor(rec["aia_stack"], dtype=torch.float32)  # (7, 512, 512)
        xs.append(resize_stack(stack, img))  # (7, img, img)
        ys.append(normalize_log_sxr(_scalar_sxr(rec["sxr_value"])))
        names.append(str(rec.get("filename", "")))

    n = len(xs)
    assert n == subsample_n, f"materialized {n} records, expected {subsample_n}"

    x = torch.stack(xs)  # (n, 7, img, img)
    y = torch.tensor(ys, dtype=torch.float32)  # (n,)

    n_val = min(max(round(n * val_frac), 1), n - 1) if n > 1 else 0
    n_train = n - n_val
    return {
        "x_train": x[:n_train],
        "y_train": y[:n_train],
        "x_val": x[n_train:],
        "y_val": y[n_train:],
        "filenames_train": names[:n_train],
        "filenames_val": names[n_train:],
        "is_streaming": bool(streaming),
        "n_materialized": n,
        "img": img,
    }


__all__ = [
    "DATASET_REPO",
    "EUV_CHANNELS",
    "LOG_SXR_MIN",
    "LOG_SXR_MAX",
    "normalize_log_sxr",
    "denormalize_log_sxr",
    "resize_stack",
    "load_foxes_subsample",
]
