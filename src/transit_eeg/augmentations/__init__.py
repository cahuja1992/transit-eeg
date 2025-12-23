"""
TRANSIT-EEG Augmentation Module

This module contains the Individualised Diffusion Probabilistic Model (IDPM)
for subject-specific EEG data augmentation.
"""

from .idpm import IDPM, create_idpm_model
from .ddpm import DenoiseDiffusion, ArcMarginHead
from .unet import UNet
from .helpers import gather, linear_beta_schedule, cosine_beta_schedule

__all__ = [
    'IDPM',
    'create_idpm_model',
    'DenoiseDiffusion',
    'ArcMarginHead',
    'UNet',
    'gather',
    'linear_beta_schedule',
    'cosine_beta_schedule',
]
