"""
Unit tests for data loaders and datasets.

Tests cover:
- SEED dataset loader
- Data preprocessing
- Feature extraction
- Differential entropy calculation
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transit_eeg.differential_entropy import (
    ButterBandpassFilter,
    BandSignalSplitter,
    BandDifferentialEntropyCalculator,
    EEGGridProjector
)
from transit_eeg.constants import (
    format_channel_location_dict,
    SEED_CHANNEL_LIST,
    PHYAAT_CHANNEL_LIST,
    CHANNEL_LOCATION_10_20
)


class TestButterBandpassFilter:
    """Test Butterworth bandpass filter."""
    
    def test_filter_initialization(self):
        """Test filter initialization."""
        filt = ButterBandpassFilter(low_cut=4, high_cut=40, fs=128, order=5)
        
        assert filt.low_cut == 4
        assert filt.high_cut == 40
        assert filt.fs == 128
        assert filt.order == 5
    
    def test_filter_apply(self):
        """Test applying filter to signal."""
        filt = ButterBandpassFilter(4, 40, 128)
        signal = np.random.randn(1000)
        
        filtered = filt.apply(signal)
        
        assert filtered.shape == signal.shape
        assert not np.array_equal(filtered, signal)
    
    def test_filter_2d_signal(self):
        """Test filtering 2D signal (multiple channels)."""
        filt = ButterBandpassFilter(4, 40, 128)
        signal = np.random.randn(32, 1000)  # 32 channels, 1000 samples
        
        filtered = filt.apply(signal)
        
        assert filtered.shape == signal.shape


class TestBandSignalSplitter:
    """Test frequency band splitting."""
    
    def test_splitter_initialization(self):
        """Test splitter initialization with default bands."""
        splitter = BandSignalSplitter(sampling_rate=128)
        
        assert splitter.sampling_rate == 128
        assert 'theta' in splitter.band_dict
        assert 'alpha' in splitter.band_dict
        assert 'beta' in splitter.band_dict
        assert 'gamma' in splitter.band_dict
    
    def test_splitter_custom_bands(self):
        """Test splitter with custom frequency bands."""
        custom_bands = {
            'delta': [1, 4],
            'theta': [4, 8],
            'alpha': [8, 13]
        }
        splitter = BandSignalSplitter(band_dict=custom_bands)
        
        assert len(splitter.band_dict) == 3
        assert splitter.band_dict['delta'] == [1, 4]
    
    def test_splitter_apply(self):
        """Test applying band splitter to signal."""
        splitter = BandSignalSplitter(sampling_rate=128)
        signal = np.random.randn(32, 1000)  # 32 channels
        
        bands = splitter.apply(signal)
        
        # Should have 4 frequency bands (theta, alpha, beta, gamma)
        assert bands.shape[0] == 4
        assert bands.shape[1:] == signal.shape
    
    def test_splitter_preserves_shape(self):
        """Test that band splitting preserves channel and time dimensions."""
        splitter = BandSignalSplitter()
        signal = np.random.randn(62, 500)
        
        bands = splitter.apply(signal)
        
        assert bands.shape[1] == 62  # Channels preserved
        assert bands.shape[2] == 500  # Time samples preserved


class TestBandDifferentialEntropyCalculator:
    """Test differential entropy calculation."""
    
    def test_entropy_calculation(self):
        """Test calculating differential entropy."""
        calculator = BandDifferentialEntropyCalculator()
        signal = np.random.randn(4, 32, 1000)  # 4 bands, 32 channels, 1000 samples
        
        entropy = calculator.apply(signal)
        
        assert isinstance(entropy, (float, np.floating))
        assert not np.isnan(entropy)
    
    def test_entropy_different_signals(self):
        """Test that different signals produce different entropy values."""
        calculator = BandDifferentialEntropyCalculator()
        
        # Signal with high variance
        signal1 = np.random.randn(4, 32, 1000) * 10
        entropy1 = calculator.apply(signal1)
        
        # Signal with low variance
        signal2 = np.random.randn(4, 32, 1000) * 0.1
        entropy2 = calculator.apply(signal2)
        
        assert entropy1 != entropy2
        assert entropy1 > entropy2  # Higher variance should have higher entropy
    
    def test_entropy_constant_signal(self):
        """Test entropy of constant signal."""
        calculator = BandDifferentialEntropyCalculator()
        signal = np.ones((4, 32, 1000))
        
        entropy = calculator.apply(signal)
        
        # Constant signal should have very low (negative infinity) entropy
        assert entropy < 0 or np.isneginf(entropy)


class TestEEGGridProjector:
    """Test EEG grid projection."""
    
    def test_projector_initialization(self):
        """Test projector initialization."""
        channel_location_dict = format_channel_location_dict(
            SEED_CHANNEL_LIST, 
            CHANNEL_LOCATION_10_20
        )
        projector = EEGGridProjector(channel_location_dict)
        
        assert projector.channel_location_dict is not None
        assert len(projector.channel_location_dict) == len(SEED_CHANNEL_LIST)
    
    def test_projector_apply(self):
        """Test applying grid projection."""
        channel_location_dict = format_channel_location_dict(
            PHYAAT_CHANNEL_LIST,
            CHANNEL_LOCATION_10_20
        )
        projector = EEGGridProjector(channel_location_dict)
        
        # Create mock 2D grid (9x9 as per CHANNEL_LOCATION_10_20)
        eeg_grid = np.random.randn(9, 9, 100)  # 9x9 spatial, 100 time samples
        
        projected = projector.apply(eeg_grid)
        
        assert projected.shape[0] == len(PHYAAT_CHANNEL_LIST)
        assert projected.shape[1] == 100  # Time samples preserved


class TestChannelConstants:
    """Test channel location constants."""
    
    def test_seed_channel_list(self):
        """Test SEED channel list."""
        assert len(SEED_CHANNEL_LIST) == 62
        assert 'FP1' in SEED_CHANNEL_LIST
        assert 'FZ' in SEED_CHANNEL_LIST
        assert 'OZ' in SEED_CHANNEL_LIST
    
    def test_phyaat_channel_list(self):
        """Test PhyAat channel list."""
        assert len(PHYAAT_CHANNEL_LIST) == 14
        assert 'AF3' in PHYAAT_CHANNEL_LIST
        assert 'T7' in PHYAAT_CHANNEL_LIST
        assert 'AF4' in PHYAAT_CHANNEL_LIST
    
    def test_channel_location_10_20(self):
        """Test 10-20 system channel locations."""
        assert len(CHANNEL_LOCATION_10_20) == 9  # 9 rows
        assert all(len(row) == 9 for row in CHANNEL_LOCATION_10_20)  # 9 columns each
    
    def test_format_channel_location_dict(self):
        """Test formatting channel location dictionary."""
        channel_dict = format_channel_location_dict(
            SEED_CHANNEL_LIST,
            CHANNEL_LOCATION_10_20
        )
        
        assert len(channel_dict) == 62
        assert all(isinstance(loc, (list, type(None))) for loc in channel_dict.values())
        
        # Check specific channels
        assert channel_dict['FP1'] is not None
        assert channel_dict['FZ'] is not None
    
    def test_format_channel_location_missing_channels(self):
        """Test handling of channels not in 10-20 system."""
        custom_channels = ['FP1', 'FZ', 'CUSTOM_CHANNEL']
        channel_dict = format_channel_location_dict(
            custom_channels,
            CHANNEL_LOCATION_10_20
        )
        
        assert channel_dict['CUSTOM_CHANNEL'] is None


class TestFeatureExtractionPipeline:
    """Test complete feature extraction pipeline."""
    
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Generate mock EEG data
        fs = 128
        duration = 2  # seconds
        n_channels = 14
        eeg_data = np.random.randn(n_channels, fs * duration)
        
        # Step 1: Bandpass filter
        bandpass = ButterBandpassFilter(4, 40, fs)
        filtered = bandpass.apply(eeg_data)
        
        # Step 2: Split into frequency bands
        splitter = BandSignalSplitter(sampling_rate=fs)
        bands = splitter.apply(filtered)
        
        # Step 3: Calculate differential entropy
        entropy_calc = BandDifferentialEntropyCalculator()
        entropy = entropy_calc.apply(bands)
        
        assert bands.shape[0] == 4  # 4 frequency bands
        assert bands.shape[1] == n_channels
        assert isinstance(entropy, (float, np.floating))
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces consistent results."""
        fs = 128
        eeg_data = np.random.randn(14, 256)
        
        # Run pipeline twice
        results = []
        for _ in range(2):
            bandpass = ButterBandpassFilter(4, 40, fs)
            filtered = bandpass.apply(eeg_data)
            
            splitter = BandSignalSplitter(sampling_rate=fs)
            bands = splitter.apply(filtered)
            
            entropy_calc = BandDifferentialEntropyCalculator()
            entropy = entropy_calc.apply(bands)
            
            results.append(entropy)
        
        # Results should be identical
        assert results[0] == results[1]


class TestDataLoaderEdgeCases:
    """Test edge cases for data loading."""
    
    def test_single_channel(self):
        """Test processing single channel."""
        bandpass = ButterBandpassFilter(4, 40, 128)
        signal = np.random.randn(256)
        
        filtered = bandpass.apply(signal)
        
        assert filtered.shape == signal.shape
    
    def test_short_signal(self):
        """Test processing very short signal."""
        bandpass = ButterBandpassFilter(4, 40, 128)
        signal = np.random.randn(10, 50)  # Only 50 samples
        
        filtered = bandpass.apply(signal)
        
        assert filtered.shape == signal.shape
    
    def test_high_dimensional_data(self):
        """Test processing high dimensional data."""
        splitter = BandSignalSplitter()
        signal = np.random.randn(128, 5000)  # Many channels and samples
        
        bands = splitter.apply(signal)
        
        assert bands.shape[1] == 128
        assert bands.shape[2] == 5000


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_invalid_sampling_rate(self):
        """Test filter with invalid sampling rate."""
        with pytest.raises(Exception):
            filt = ButterBandpassFilter(4, 40, -128)
    
    def test_invalid_frequency_range(self):
        """Test filter with invalid frequency range."""
        with pytest.raises(Exception):
            # High cut lower than low cut
            filt = ButterBandpassFilter(40, 4, 128)
            signal = np.random.randn(100)
            filt.apply(signal)
    
    def test_empty_channel_list(self):
        """Test formatting empty channel list."""
        channel_dict = format_channel_location_dict([], CHANNEL_LOCATION_10_20)
        
        assert len(channel_dict) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
