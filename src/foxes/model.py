"""mini-FOXES `ViTLocal` -- a from-scratch, mechanism-faithful miniature of the FOXES ViT.

Adapted from (and citing) the public FOXES reference implementation
`forecasting/models/vit_patch_model_local.py` (github.com/griffin-goodwin/FOXES;
Goodwin et al. 2026, ApJ, DOI 10.3847/1538-4357/ae5e4a). The distinctive FOXES signatures
reproduced here:

  * 7-channel `Conv2d` patch embedding at 8x8 stride (no RGB, trained fresh on EUV channels).
  * A 9x9 *inverted* masked-attention block (`mask_mode="inverted"`, `local_window=9`, r=4):
    the 9x9 local patch neighborhood is BLOCKED so every patch attends only to *distant*
    patches -- the FOXES "non-local" mechanism that emphasises global context. ALL `depth`
    encoder blocks are this block; there is no alternation and no standard-attention subset.
  * NO CLS / class token -- global context comes from the masked attention, not a special
    aggregation token (token count is exactly N = number of patches).
  * A per-patch linear regression head `Linear(embed, 1)`: one scalar per patch, and the
    scalars SUM to the global SXR prediction. This is the intrinsic spatial attribution map
    (the prediction, decomposed) -- the primary XAI artifact.

Attention is deliberately materializable: this module implements attention *eagerly*
(explicit softmax over QK^T, no SDPA / Flash / fused path), so the per-block (B, H, N, N)
weights can be captured by `src/foxes/attention.py` for the XAI sanity figure. timm's
`set_fused_attn(False)` is called at import time as belt-and-suspenders in case timm
primitives are ever introduced.

`aux_mask` hook: an optional auxiliary active-region mask path is stubbed here for the real
SuryaBench AR-segmentation masks (Phase 5 documents the drop-in); in Phase 1 it is an
extra-channel placeholder so the signature is pinned and the path is not dead code.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:  # timm is an optional ([foxes]) dependency; only used to force the eager attn path.
    import timm

    timm.layers.set_fused_attn(False)
except Exception:  # pragma: no cover - timm absent is fine, this module is self-contained.
    pass


class InvertedAttentionBlock(nn.Module):
    """A pre-norm transformer encoder block with eager multi-head self-attention + an
    inverted (non-local) attention mask.

    The attention is computed explicitly (softmax over scaled QK^T) rather than through
    `F.scaled_dot_product_attention`, so the (B, H, N, N) weight matrix exists and can be
    read by a forward hook. The most recent attention weights are also stashed on
    ``self.attn_weights`` after every forward for convenience.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Most recent attention weights (B, H, N, N); populated on every forward so a hook
        # (or a direct read) can extract them for XAI. Not a buffer (varies with batch).
        self.attn_weights: Tensor | None = None

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """x: (B, N, D). attn_mask: (N, N) boolean, True where attention is FORBIDDEN."""
        b, n, d = x.shape
        h = self.num_heads

        residual = x
        y = self.norm1(x)
        qkv = self.qkv(y).reshape(b, n, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if attn_mask is not None:
            # attn_mask True == blocked. Broadcast (N, N) over (B, H, N, N).
            attn = attn.masked_fill(attn_mask.view(1, 1, n, n), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attn_weights = attn  # materialized weights for XAI extraction

        out = (attn @ v).transpose(1, 2).reshape(b, n, d)  # (B, N, D)
        out = self.proj_drop(self.proj(out))
        x = residual + out

        x = x + self.mlp(self.norm2(x))
        return x


class ViTLocal(nn.Module):
    """From-scratch FOXES-style ViT: multi-channel EUV stack -> per-patch SXR contributions.

    forward(x) -> (global_pred (B,), per_patch (B, N)) with per_patch.sum(1) == global_pred.
    """

    def __init__(
        self,
        in_chans: int = 7,
        img: int = 128,
        patch: int = 8,
        embed: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp: int = 1024,
        local_window: int = 9,
        aux_mask_chans: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img % patch == 0, "image size must be divisible by patch size"
        self.in_chans = in_chans
        self.aux_mask_chans = aux_mask_chans
        self.img = img
        self.patch = patch
        self.embed = embed
        self.depth = depth
        self.heads = heads
        self.local_window = local_window

        self.grid_h = img // patch
        self.grid_w = img // patch
        self.num_patches = self.grid_h * self.grid_w  # N (no CLS token added)

        # Patch embed over (in_chans + aux_mask_chans). The aux channels are the stubbed
        # Surya AR-mask drop-in (Phase 5); aux_mask_chans=0 in the Phase-1 default.
        self.patch_embed = nn.Conv2d(
            in_chans + aux_mask_chans, embed, kernel_size=patch, stride=patch
        )
        # Learned absolute positional embedding, one per patch (NO CLS token entry).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Every encoder block is an InvertedAttentionBlock (faithful to FOXES: no standard
        # attention layer, no alternation).
        self.blocks = nn.ModuleList(
            [InvertedAttentionBlock(embed, heads, mlp, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed)

        # Per-patch linear regression head -> one scalar per patch. The scalars SUM to the
        # global SXR prediction (intrinsic spatial attribution).
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed), nn.Linear(embed, 1))

        # Precompute and register the (N, N) inverted attention mask (r = local_window // 2).
        mask = self.build_inverted_mask(self.grid_h, self.grid_w, r=local_window // 2)
        self.register_buffer("attn_mask", mask, persistent=False)

    def build_inverted_mask(self, grid_h: int, grid_w: int, r: int = 4) -> Tensor:
        """Return an (N, N) boolean mask, True where attention is FORBIDDEN.

        Reproduces FOXES `mask_mode="inverted"`: the (2r+1)x(2r+1) local patch neighborhood
        is blocked, so each patch attends only to distant patches. With local_window=9, r=4
        gives the paper's 9x9 neighborhood.
        """
        n = grid_h * grid_w
        rows = torch.arange(n) // grid_w
        cols = torch.arange(n) % grid_w
        dr = (rows[:, None] - rows[None, :]).abs()
        dc = (cols[:, None] - cols[None, :]).abs()
        local = (dr <= r) & (dc <= r)  # True inside the 9x9 neighborhood (incl. diagonal)
        return local  # inverted: local == blocked

    def forward(self, x: Tensor, aux_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """x: (B, in_chans, img, img). aux_mask: optional (B, aux_mask_chans, img, img).

        Returns (global_pred (B,), per_patch (B, N)) with per_patch.sum(1) == global_pred.
        """
        if self.aux_mask_chans:
            if aux_mask is None:
                aux_mask = x.new_zeros(x.shape[0], self.aux_mask_chans, x.shape[2], x.shape[3])
            x = torch.cat([x, aux_mask], dim=1)

        p = self.patch_embed(x)  # (B, embed, grid_h, grid_w)
        p = p.flatten(2).transpose(1, 2)  # (B, N, embed)
        p = self.pos_drop(p + self.pos_embed)

        for blk in self.blocks:
            p = blk(p, self.attn_mask)
        p = self.norm(p)

        per_patch = self.mlp_head(p).squeeze(-1)  # (B, N) intrinsic attribution
        global_pred = per_patch.sum(1)  # (B,) global SXR (sum of patch contributions)
        return global_pred, per_patch


__all__ = ["InvertedAttentionBlock", "ViTLocal"]
