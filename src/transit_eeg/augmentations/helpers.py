"""
Helper functions for IDPM diffusion process.
"""

import torch
import torch.nn.functional as F


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather constants for time step t and reshape to feature map shape.
    
    Args:
        consts: Tensor of constants with shape [T]
        t: Time step indices with shape [batch_size]
        
    Returns:
        Gathered constants reshaped to [batch_size, 1, 1, 1]
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def extract(a, t, x_shape):
    """
    Extract values from tensor a at indices t and reshape to match x_shape.
    
    Args:
        a: Tensor to extract from
        t: Time step indices
        x_shape: Target shape
        
    Returns:
        Extracted and reshaped tensor
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule for beta values.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Tensor of beta values
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values as proposed in
    'Improved Denoising Diffusion Probabilistic Models'.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small
        
    Returns:
        Tensor of beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Quadratic schedule for beta values.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Tensor of beta values
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Sigmoid schedule for beta values.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Tensor of beta values
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
