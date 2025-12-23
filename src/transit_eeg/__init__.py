"""
TRANSIT-EEG: Transfer and Robust Adaptation for New Subjects in EEG Technology

Official implementation of the TRANSIT-EEG framework for cross-subject EEG classification.

Main Components:
- IDPM: Individualised Diffusion Probabilistic Model for data augmentation
- SOGAT: Self-Organizing Graph Attention Transformer for classification
- LoRA: Low-Rank Adaptation for efficient finetuning

Authors: Chirag Ahuja, Divyashikha Sethia
"""

__version__ = '1.0.0'
__author__ = 'Chirag Ahuja, Divyashikha Sethia'

from . import augmentations
from . import model
from . import datasets
from . import utils

__all__ = [
    'augmentations',
    'model',
    'datasets',
    'utils',
]
