"""
Individualised Diffusion Probabilistic Model (IDPM) for EEG Data Augmentation.

This module implements the IDPM as described in the TRANSIT-EEG paper.
IDPM decomposes EEG signals into cleaned components and subject-specific artifacts,
enabling high-quality synthetic data generation for augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from tqdm import tqdm

from .unet import UNet
from .ddpm import ArcMarginHead, DenoiseDiffusion
from .helpers import gather, linear_beta_schedule


class IDPM(nn.Module):
    """
    Individualised Diffusion Probabilistic Model for subject-specific EEG augmentation.
    
    The IDPM separates noise from subject-specific informative signals during the
    denoising process, unlike vanilla DDPM which only denoises without separate branches.
    
    Args:
        n_steps: Number of diffusion timesteps (default: 1000)
        channels: Number of EEG channels (default: 62 for SEED)
        window_size: Time window size for each sample (default: 265 for SEED)
        stack_size: Number of frequency bands (default: 5)
        device: Device to run on ('cuda' or 'cpu')
        beta_schedule: Type of beta schedule ('linear', 'cosine', etc.)
    """
    
    def __init__(
        self,
        n_steps: int = 1000,
        channels: int = 62,
        window_size: int = 265,
        stack_size: int = 5,
        device: str = 'cuda',
        beta_schedule: str = 'linear',
        arc_in: int = 200,
        arc_out: int = 15,
        learning_rate: float = 2e-5
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.channels = channels
        self.window_size = window_size
        self.stack_size = stack_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.eps_model_clean = UNet(
            in_channels=1,
            out_channels=1,
            channels=64,
            n_res_blocks=2
        ).to(self.device)
        
        self.eps_model_noise = UNet(
            in_channels=1,
            out_channels=1,
            channels=64,
            n_res_blocks=2
        ).to(self.device)
        
        # Subject classifier for arc-margin loss
        self.subject_classifier = ArcMarginHead(
            in_features=arc_in,
            out_features=arc_out
        ).to(self.device)
        
        # Initialize diffusion process
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model_clean,
            n_steps=n_steps,
            device=self.device
        )
        
        self.learning_rate = learning_rate
        
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: gradually add noise to signal.
        
        Args:
            x0: Clean EEG signal [batch_size, channels, height, width]
            t: Timestep [batch_size]
            noise: Optional noise tensor (generated if None)
            
        Returns:
            Tuple of (noisy signal xt, noise epsilon)
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        # Get alpha values for timestep t
        alpha_t = self.diffusion.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1)
        
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        
        return xt, noise
    
    def reverse_diffusion(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor,
        subject_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse diffusion process: denoise signal into clean and noise components.
        
        Args:
            xt: Noisy signal at timestep t
            t: Current timestep
            subject_id: Subject identifier for conditioning
            
        Returns:
            Tuple of (cleaned signal, subject-specific noise)
        """
        # Predict clean component
        eps_clean = self.eps_model_clean(xt, t)
        
        # Predict subject-specific noise component
        eps_noise = self.eps_model_noise(xt, t)
        
        return eps_clean, eps_noise
    
    def compute_losses(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        eps_true: torch.Tensor,
        subject_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the three loss functions: Reverse Loss, Orthogonal Loss, Arc-Margin Loss.
        
        Args:
            x0: Original clean signal
            xt: Noisy signal at timestep t
            t: Timestep
            eps_true: True noise
            subject_ids: Subject identifiers
            
        Returns:
            Tuple of (total_loss, reverse_loss, orthogonal_loss, arc_margin_loss)
        """
        # Get predictions
        eps_clean, eps_noise = self.reverse_diffusion(xt, t, subject_ids)
        
        # 1. Reverse Loss (L_r): Reconstruction loss for clean signal
        alpha_t = self.diffusion.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1)
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * eps_clean) / torch.sqrt(alpha_t)
        reverse_loss = F.mse_loss(x0_pred, x0)
        
        # 2. Orthogonal Loss (L_o): Ensure clean and noise components are orthogonal
        eps_clean_flat = eps_clean.flatten(start_dim=1)
        eps_noise_flat = eps_noise.flatten(start_dim=1)
        orthogonal_loss = torch.norm(
            torch.bmm(eps_clean_flat.unsqueeze(1), eps_noise_flat.unsqueeze(2)),
            p='fro'
        )
        
        # 3. Arc-Margin Loss (L_arc): Subject discriminability
        # Extract features from noise component for subject classification
        noise_features = eps_noise.mean(dim=(2, 3))  # Pool spatial dimensions
        arc_margin_loss = self.subject_classifier(noise_features, subject_ids)
        
        # Combined loss with weights (as per paper)
        lambda_r = 1.0
        lambda_o = 0.1
        lambda_arc = 0.5
        
        total_loss = (lambda_r * reverse_loss + 
                     lambda_o * orthogonal_loss + 
                     lambda_arc * arc_margin_loss)
        
        return total_loss, reverse_loss, orthogonal_loss, arc_margin_loss
    
    def train_step(
        self,
        x0: torch.Tensor,
        subject_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        Perform one training step.
        
        Args:
            x0: Batch of clean EEG signals
            subject_ids: Batch of subject identifiers
            optimizer: Optimizer
            
        Returns:
            Dictionary of losses
        """
        optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x0.shape[0],), device=self.device)
        
        # Forward diffusion
        xt, eps_true = self.forward_diffusion(x0, t)
        
        # Compute losses
        total_loss, reverse_loss, ortho_loss, arc_loss = self.compute_losses(
            x0, xt, t, eps_true, subject_ids
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reverse_loss': reverse_loss.item(),
            'orthogonal_loss': ortho_loss.item(),
            'arc_margin_loss': arc_loss.item()
        }
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        subject_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate synthetic EEG samples through reverse diffusion.
        
        Args:
            num_samples: Number of samples to generate
            subject_id: Subject ID for conditioning (optional)
            
        Returns:
            Generated samples [num_samples, channels, height, width]
        """
        # Start from pure noise
        xt = torch.randn(
            num_samples, 1, self.channels, self.window_size,
            device=self.device
        )
        
        # Iteratively denoise
        for t in tqdm(reversed(range(self.n_steps)), desc='Sampling', total=self.n_steps):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict clean and noise components
            eps_clean, eps_noise = self.reverse_diffusion(xt, t_batch, subject_id)
            
            # Combine components (weighted combination)
            eps_combined = eps_clean + eps_noise
            
            # Compute x_{t-1}
            alpha_t = self.diffusion.alpha_bar[t]
            alpha_t_prev = self.diffusion.alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
            
            beta_t = 1 - (alpha_t / alpha_t_prev)
            
            # Mean of q(x_{t-1} | x_t, x_0)
            x0_pred = (xt - torch.sqrt(1 - alpha_t) * eps_combined) / torch.sqrt(alpha_t)
            mean = (torch.sqrt(alpha_t_prev) * beta_t * x0_pred + 
                   torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) * xt) / (1 - alpha_t)
            
            # Add noise if not final step
            if t > 0:
                noise = torch.randn_like(xt)
                sigma_t = torch.sqrt(beta_t)
                xt = mean + sigma_t * noise
            else:
                xt = mean
        
        return xt
    
    def augment_subject_data(
        self,
        base_samples: torch.Tensor,
        subject_id: int,
        aug_factor: float = 1.5
    ) -> torch.Tensor:
        """
        Augment data for a specific subject using IDPM.
        
        Args:
            base_samples: Original samples for the subject
            subject_id: Subject identifier
            aug_factor: Augmentation factor (e.g., 1.5 means 50% more samples)
            
        Returns:
            Augmented dataset including original samples
        """
        num_to_generate = int(len(base_samples) * (aug_factor - 1))
        
        if num_to_generate > 0:
            synthetic_samples = self.sample(num_samples=num_to_generate, subject_id=subject_id)
            return torch.cat([base_samples, synthetic_samples], dim=0)
        else:
            return base_samples


def create_idpm_model(config: dict) -> IDPM:
    """
    Factory function to create IDPM model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized IDPM model
    """
    return IDPM(
        n_steps=config.get('n_steps', 1000),
        channels=config.get('channels', 62),
        window_size=config.get('window_size', 265),
        stack_size=config.get('stack_size', 5),
        device=config.get('device', 'cuda'),
        beta_schedule=config.get('beta_schedule', 'linear'),
        arc_in=config.get('arc_in', 200),
        arc_out=config.get('arc_out', 15),
        learning_rate=config.get('learning_rate', 2e-5)
    )
