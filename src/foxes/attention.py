"""Eager-mode attention extraction for mini-FOXES XAI.

The single most common failure mode for attention-map XAI is that fused / SDPA / Flash
attention never materializes the (B, H, N, N) weight matrix. `ViTLocal` therefore computes
attention eagerly (explicit softmax over QK^T) and each `InvertedAttentionBlock` stashes its
most recent weights on `.attn_weights`. This helper runs a forward pass under a set of
forward hooks and returns `{block_idx: (B, H, N, N)}`.

If timm primitives are ever introduced into the model, `set_fused_attn(False)` must have been
called first; `assert_eager()` checks that and raises a clear error otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import InvertedAttentionBlock, ViTLocal


def assert_eager() -> None:
    """Assert timm's fused-attention path is disabled (no-op if timm is absent).

    Raises AssertionError if timm is importable but still has fused attention enabled, since
    that would silently yield empty/None attention weights when timm attn primitives are used.
    """
    try:
        import timm
    except Exception:
        return  # timm not installed -> ViTLocal's hand-rolled eager attention is used.
    fused = getattr(timm.layers, "use_fused_attn", None)
    if callable(fused):
        assert not fused(), "timm fused attention is ON; call timm.layers.set_fused_attn(False)"


@torch.no_grad()
def extract_attention(
    model: ViTLocal, x: Tensor, aux_mask: Tensor | None = None
) -> dict[int, Tensor]:
    """Run a forward pass with hooks and return {block_idx: (B, H, N, N)} attention weights.

    Uses forward hooks on each `InvertedAttentionBlock` to read the `.attn_weights` the block
    materialized during the forward (analogous to capturing module_out[1] from an eager
    `nn.MultiheadAttention(need_weights=True, average_attn_weights=False)`).
    """
    assert_eager()
    model.eval()

    captured: dict[int, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(idx: int):
        def hook(module: nn.Module, _inp, _out):
            w = getattr(module, "attn_weights", None)
            if w is not None:
                captured[idx] = w.detach()

        return hook

    block_idx = 0
    for module in model.modules():
        if isinstance(module, InvertedAttentionBlock):
            handles.append(module.register_forward_hook(make_hook(block_idx)))
            block_idx += 1

    try:
        model(x, aux_mask=aux_mask)
    finally:
        for h in handles:
            h.remove()

    return captured


__all__ = ["assert_eager", "extract_attention"]
