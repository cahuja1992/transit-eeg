"""
TRANSIT-EEG Model Module

This module contains the SOGAT (Self-Organizing Graph Attention Transformer)
model and related components for EEG signal classification.
"""

from .sogat import SOGAT
from .sognn import SOGNN
from .modules import (
    DenseGATConv,
    SOGC,
    AdapterLayer,
    LowRankAdapterLayer,
    glorot,
    zeros
)

__all__ = [
    'SOGAT',
    'SOGNN',
    'DenseGATConv',
    'SOGC',
    'AdapterLayer',
    'LowRankAdapterLayer',
    'glorot',
    'zeros',
]
