"""Pluvian — multi-source short-range precipitation nowcasting model.

Public API:
  - ``Pluvian``: main model (encoder + cross-attn fusion + decoder + dual heads)
  - building blocks exported for unit testing and ablation use
"""
from .pluvian import Pluvian
from .encoder import RadarEncoder, Era5Encoder, SparseTokenEncoder
from .fusion import CrossAttentionFusion
from .decoder import PluvianDecoder

__all__ = [
    "Pluvian",
    "RadarEncoder",
    "Era5Encoder",
    "SparseTokenEncoder",
    "CrossAttentionFusion",
    "PluvianDecoder",
]
