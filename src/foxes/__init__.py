"""mini-FOXES: a small, real, mechanism-faithful miniature of the FOXES ViT.

A from-scratch Vision Transformer (multi-channel SDO/AIA EUV stack -> GOES SXR flux
regression) plus its spatial Explainable-AI surface, built on a dedicated branch as a
proof-of-mechanism showcase. NOT part of the anomaly-detection comparison harness in
src/inference/evaluate.py (different task, different metric -- see the design discussion).

Reference architecture: the public FOXES `vit_patch_model_local.py`
(github.com/griffin-goodwin/FOXES), Goodwin et al. 2026, ApJ, DOI 10.3847/1538-4357/ae5e4a.
"""
