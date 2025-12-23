"""
Pytest configuration file for TRANSIT-EEG tests.

This file configures pytest for the test suite, including:
- Test discovery settings
- Coverage configuration
- Fixtures
- Markers
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(scope="session")
def device():
    """Provide device for testing (CPU by default)."""
    return torch.device('cpu')


@pytest.fixture
def seed_channels():
    """Provide SEED dataset channel count."""
    return 62


@pytest.fixture
def phyaat_channels():
    """Provide PhyAat dataset channel count."""
    return 14


@pytest.fixture
def num_classes():
    """Provide number of classes."""
    return 3


@pytest.fixture
def sample_eeg_data(seed_channels):
    """Generate sample EEG data for testing."""
    batch_size = 2
    freq_bands = 5
    time_samples = 265
    
    return torch.randn(batch_size, 1, seed_channels, time_samples)


@pytest.fixture
def sample_labels(num_classes):
    """Generate sample labels for testing."""
    batch_size = 2
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for saving checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# Hooks for test reporting
def pytest_report_header(config):
    """Add custom header to pytest output."""
    return [
        "TRANSIT-EEG Test Suite",
        f"PyTorch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"NumPy version: {np.__version__}"
    ]


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
