"""
Unit tests for IDPM (Individualised Diffusion Probabilistic Model).

Tests cover:
- Forward diffusion process
- Reverse diffusion process
- Loss functions (Reverse, Orthogonal, Arc-Margin)
- Sampling and generation
- Subject-specific augmentation
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transit_eeg.augmentations.idpm import IDPM, create_idpm_model
from transit_eeg.augmentations.helpers import (
    gather, 
    linear_beta_schedule, 
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule
)


class TestHelperFunctions:
    """Test helper functions for diffusion process."""
    
    def test_gather_function(self):
        """Test gather function for extracting timestep values."""
        consts = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        t = torch.tensor([0, 2, 4])
        
        result = gather(consts, t)
        
        assert result.shape == (3, 1, 1, 1)
        assert result[0, 0, 0, 0] == 1.0
        assert result[1, 0, 0, 0] == 3.0
        assert result[2, 0, 0, 0] == 5.0
    
    def test_linear_beta_schedule(self):
        """Test linear beta schedule generation."""
        timesteps = 1000
        betas = linear_beta_schedule(timesteps)
        
        assert betas.shape == (timesteps,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
        assert betas[0] < betas[-1]  # Should be increasing
    
    def test_cosine_beta_schedule(self):
        """Test cosine beta schedule generation."""
        timesteps = 1000
        betas = cosine_beta_schedule(timesteps)
        
        assert betas.shape == (timesteps,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
    
    def test_quadratic_beta_schedule(self):
        """Test quadratic beta schedule generation."""
        timesteps = 1000
        betas = quadratic_beta_schedule(timesteps)
        
        assert betas.shape == (timesteps,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
    
    def test_sigmoid_beta_schedule(self):
        """Test sigmoid beta schedule generation."""
        timesteps = 1000
        betas = sigmoid_beta_schedule(timesteps)
        
        assert betas.shape == (timesteps,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)


class TestIDPMInitialization:
    """Test IDPM model initialization."""
    
    def test_default_initialization(self):
        """Test IDPM initialization with default parameters."""
        model = IDPM(device='cpu')
        
        assert model.n_steps == 1000
        assert model.channels == 62
        assert model.window_size == 265
        assert model.stack_size == 5
        assert model.device.type == 'cpu'
    
    def test_custom_initialization(self):
        """Test IDPM initialization with custom parameters."""
        model = IDPM(
            n_steps=500,
            channels=14,
            window_size=384,
            stack_size=4,
            device='cpu'
        )
        
        assert model.n_steps == 500
        assert model.channels == 14
        assert model.window_size == 384
        assert model.stack_size == 4
    
    def test_factory_function(self):
        """Test IDPM creation via factory function."""
        config = {
            'n_steps': 800,
            'channels': 32,
            'window_size': 200,
            'device': 'cpu'
        }
        
        model = create_idpm_model(config)
        
        assert model.n_steps == 800
        assert model.channels == 32
        assert model.window_size == 200


class TestIDPMForwardDiffusion:
    """Test forward diffusion process."""
    
    @pytest.fixture
    def idpm_model(self):
        """Create IDPM model for testing."""
        return IDPM(n_steps=100, channels=4, window_size=16, device='cpu')
    
    def test_forward_diffusion_shape(self, idpm_model):
        """Test forward diffusion output shape."""
        batch_size = 2
        x0 = torch.randn(batch_size, 1, 4, 16)
        t = torch.randint(0, 100, (batch_size,))
        
        xt, noise = idpm_model.forward_diffusion(x0, t)
        
        assert xt.shape == x0.shape
        assert noise.shape == x0.shape
    
    def test_forward_diffusion_noise(self, idpm_model):
        """Test forward diffusion with provided noise."""
        batch_size = 2
        x0 = torch.randn(batch_size, 1, 4, 16)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x0)
        
        xt, returned_noise = idpm_model.forward_diffusion(x0, t, noise)
        
        assert torch.allclose(noise, returned_noise)
    
    def test_forward_diffusion_t0(self, idpm_model):
        """Test forward diffusion at t=0 (should be close to original)."""
        x0 = torch.randn(2, 1, 4, 16)
        t = torch.zeros(2, dtype=torch.long)
        
        xt, _ = idpm_model.forward_diffusion(x0, t)
        
        # At t=0, xt should be very close to x0
        assert torch.allclose(xt, x0, atol=0.1)


class TestIDPMReverseDiffusion:
    """Test reverse diffusion process."""
    
    @pytest.fixture
    def idpm_model(self):
        """Create IDPM model for testing."""
        return IDPM(n_steps=100, channels=4, window_size=16, device='cpu')
    
    def test_reverse_diffusion_shape(self, idpm_model):
        """Test reverse diffusion output shapes."""
        batch_size = 2
        xt = torch.randn(batch_size, 1, 4, 16)
        t = torch.randint(0, 100, (batch_size,))
        
        eps_clean, eps_noise = idpm_model.reverse_diffusion(xt, t)
        
        assert eps_clean.shape == xt.shape
        assert eps_noise.shape == xt.shape
    
    def test_reverse_diffusion_deterministic(self, idpm_model):
        """Test that reverse diffusion is deterministic."""
        xt = torch.randn(2, 1, 4, 16)
        t = torch.randint(0, 100, (2,))
        
        idpm_model.eval()
        with torch.no_grad():
            eps_clean1, eps_noise1 = idpm_model.reverse_diffusion(xt, t)
            eps_clean2, eps_noise2 = idpm_model.reverse_diffusion(xt, t)
        
        assert torch.allclose(eps_clean1, eps_clean2)
        assert torch.allclose(eps_noise1, eps_noise2)


class TestIDPMLosses:
    """Test IDPM loss functions."""
    
    @pytest.fixture
    def idpm_model(self):
        """Create IDPM model for testing."""
        return IDPM(n_steps=100, channels=4, window_size=16, device='cpu')
    
    def test_loss_computation(self, idpm_model):
        """Test that losses are computed correctly."""
        batch_size = 2
        x0 = torch.randn(batch_size, 1, 4, 16)
        t = torch.randint(0, 100, (batch_size,))
        subject_ids = torch.randint(0, 5, (batch_size,))
        
        xt, eps_true = idpm_model.forward_diffusion(x0, t)
        total_loss, reverse_loss, ortho_loss, arc_loss = idpm_model.compute_losses(
            x0, xt, t, eps_true, subject_ids
        )
        
        # Check all losses are scalar tensors
        assert total_loss.ndim == 0
        assert reverse_loss.ndim == 0
        assert ortho_loss.ndim == 0
        assert arc_loss.ndim == 0
        
        # Check all losses are non-negative
        assert total_loss >= 0
        assert reverse_loss >= 0
        assert ortho_loss >= 0
    
    def test_reverse_loss_is_positive(self, idpm_model):
        """Test that reverse loss is always positive."""
        x0 = torch.randn(2, 1, 4, 16)
        t = torch.randint(0, 100, (2,))
        subject_ids = torch.randint(0, 5, (2,))
        
        xt, eps_true = idpm_model.forward_diffusion(x0, t)
        _, reverse_loss, _, _ = idpm_model.compute_losses(
            x0, xt, t, eps_true, subject_ids
        )
        
        assert reverse_loss > 0


class TestIDPMTraining:
    """Test IDPM training functionality."""
    
    @pytest.fixture
    def idpm_model(self):
        """Create IDPM model for testing."""
        return IDPM(n_steps=50, channels=4, window_size=16, device='cpu')
    
    def test_train_step(self, idpm_model):
        """Test training step execution."""
        x0 = torch.randn(2, 1, 4, 16)
        subject_ids = torch.randint(0, 5, (2,))
        optimizer = torch.optim.Adam(idpm_model.parameters(), lr=0.001)
        
        losses = idpm_model.train_step(x0, subject_ids, optimizer)
        
        assert 'total_loss' in losses
        assert 'reverse_loss' in losses
        assert 'orthogonal_loss' in losses
        assert 'arc_margin_loss' in losses
        
        assert all(isinstance(v, float) for v in losses.values())
        assert all(v >= 0 for v in losses.values())


class TestIDPMSampling:
    """Test IDPM sampling and generation."""
    
    @pytest.fixture
    def idpm_model(self):
        """Create IDPM model for testing."""
        return IDPM(n_steps=10, channels=4, window_size=16, device='cpu')  # Few steps for speed
    
    def test_sample_shape(self, idpm_model):
        """Test that sampling produces correct shape."""
        num_samples = 3
        
        samples = idpm_model.sample(num_samples=num_samples)
        
        assert samples.shape == (num_samples, 1, 4, 16)
    
    def test_sample_with_subject_id(self, idpm_model):
        """Test sampling with subject conditioning."""
        num_samples = 2
        subject_id = 0
        
        samples = idpm_model.sample(num_samples=num_samples, subject_id=subject_id)
        
        assert samples.shape == (num_samples, 1, 4, 16)
    
    def test_sample_different_each_time(self, idpm_model):
        """Test that samples are different each time."""
        samples1 = idpm_model.sample(num_samples=2)
        samples2 = idpm_model.sample(num_samples=2)
        
        assert not torch.allclose(samples1, samples2)


class TestIDPMAugmentation:
    """Test subject-specific augmentation."""
    
    @pytest.fixture
    def idpm_model(self):
        """Create IDPM model for testing."""
        return IDPM(n_steps=10, channels=4, window_size=16, device='cpu')
    
    def test_augment_subject_data_shape(self, idpm_model):
        """Test augmentation produces correct number of samples."""
        base_samples = torch.randn(5, 1, 4, 16)
        subject_id = 0
        aug_factor = 1.5
        
        augmented = idpm_model.augment_subject_data(
            base_samples, subject_id, aug_factor
        )
        
        expected_total = 5 + int(5 * (aug_factor - 1))
        assert augmented.shape[0] == expected_total
        assert augmented.shape[1:] == base_samples.shape[1:]
    
    def test_augment_with_factor_1(self, idpm_model):
        """Test augmentation with factor 1.0 returns original data."""
        base_samples = torch.randn(5, 1, 4, 16)
        subject_id = 0
        
        augmented = idpm_model.augment_subject_data(
            base_samples, subject_id, aug_factor=1.0
        )
        
        assert augmented.shape == base_samples.shape
        assert torch.allclose(augmented, base_samples)
    
    def test_augment_with_factor_2(self, idpm_model):
        """Test augmentation with factor 2.0 doubles data."""
        base_samples = torch.randn(5, 1, 4, 16)
        subject_id = 0
        
        augmented = idpm_model.augment_subject_data(
            base_samples, subject_id, aug_factor=2.0
        )
        
        assert augmented.shape[0] == 10  # 5 original + 5 synthetic


class TestIDPMEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_timesteps(self):
        """Test initialization with invalid number of timesteps."""
        with pytest.raises(Exception):
            IDPM(n_steps=-1, device='cpu')
    
    def test_invalid_channels(self):
        """Test initialization with invalid number of channels."""
        with pytest.raises(Exception):
            IDPM(channels=0, device='cpu')
    
    def test_batch_size_one(self):
        """Test that model works with batch size 1."""
        model = IDPM(n_steps=50, channels=4, window_size=16, device='cpu')
        x0 = torch.randn(1, 1, 4, 16)
        t = torch.randint(0, 50, (1,))
        
        xt, noise = model.forward_diffusion(x0, t)
        
        assert xt.shape == x0.shape


class TestIDPMIntegration:
    """Integration tests for IDPM."""
    
    def test_full_training_cycle(self):
        """Test complete training cycle."""
        model = IDPM(n_steps=10, channels=4, window_size=16, device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Simulate a few training steps
        for _ in range(3):
            x0 = torch.randn(2, 1, 4, 16)
            subject_ids = torch.randint(0, 5, (2,))
            
            losses = model.train_step(x0, subject_ids, optimizer)
            
            assert losses['total_loss'] >= 0
    
    def test_train_and_sample(self):
        """Test training followed by sampling."""
        model = IDPM(n_steps=10, channels=4, window_size=16, device='cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few steps
        x0 = torch.randn(2, 1, 4, 16)
        subject_ids = torch.randint(0, 5, (2,))
        model.train_step(x0, subject_ids, optimizer)
        
        # Sample
        samples = model.sample(num_samples=2)
        
        assert samples.shape == (2, 1, 4, 16)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
