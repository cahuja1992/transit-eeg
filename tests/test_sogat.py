"""
Unit tests for SOGAT (Self-Organizing Graph Attention Transformer).

Tests cover:
- Model initialization
- Forward pass
- Dynamic graph construction
- GAT layers
- LoRA adapters
- Training functionality
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transit_eeg.model.sogat import SOGAT
from transit_eeg.model.modules import (
    DenseGATConv,
    SOGC,
    AdapterLayer,
    LowRankAdapterLayer,
    glorot,
    zeros
)


class TestInitializationFunctions:
    """Test weight initialization functions."""
    
    def test_glorot_initialization(self):
        """Test Glorot/Xavier initialization."""
        tensor = torch.empty(10, 20)
        glorot(tensor)
        
        assert tensor.std() > 0
        assert tensor.mean().abs() < 0.1
    
    def test_zeros_initialization(self):
        """Test zeros initialization."""
        tensor = torch.randn(5, 5)
        zeros(tensor)
        
        assert torch.all(tensor == 0)
    
    def test_glorot_with_none(self):
        """Test that glorot handles None input."""
        glorot(None)  # Should not raise error
    
    def test_zeros_with_none(self):
        """Test that zeros handles None input."""
        zeros(None)  # Should not raise error


class TestAdapterLayer:
    """Test standard adapter layer."""
    
    def test_adapter_initialization(self):
        """Test adapter layer initialization."""
        adapter = AdapterLayer(
            input_channels=64,
            hidden_channels=16,
            output_channels=64
        )
        
        assert adapter.linear1.in_features == 64
        assert adapter.linear1.out_features == 16
        assert adapter.linear2.in_features == 16
        assert adapter.linear2.out_features == 64
    
    def test_adapter_forward(self):
        """Test adapter forward pass."""
        adapter = AdapterLayer(64, 16, 64, non_linear_activation=torch.nn.ReLU())
        x = torch.randn(4, 64)
        
        output = adapter(x)
        
        assert output.shape == (4, 64)
    
    def test_adapter_without_activation(self):
        """Test adapter without activation function."""
        adapter = AdapterLayer(64, 16, 64, non_linear_activation=None)
        x = torch.randn(4, 64)
        
        output = adapter(x)
        
        assert output.shape == (4, 64)


class TestLowRankAdapterLayer:
    """Test LoRA (Low-Rank Adapter) layer."""
    
    def test_lora_initialization(self):
        """Test LoRA initialization."""
        lora = LowRankAdapterLayer(
            input_channels=64,
            rank=8,
            output_channels=64
        )
        
        assert lora.rank == 8
        assert lora.lora_matrix_B.shape == (64, 8)
        assert lora.lora_matrix_A.shape == (8, 64)
    
    def test_lora_forward(self):
        """Test LoRA forward pass."""
        lora = LowRankAdapterLayer(64, 8, 64)
        x = torch.randn(4, 64)
        
        output = lora(x)
        
        assert output.shape == (4, 64)
    
    def test_lora_residual_connection(self):
        """Test that LoRA uses residual connection."""
        lora = LowRankAdapterLayer(64, 8, 64)
        
        # Initialize matrices to zero
        lora.lora_matrix_B.data.zero_()
        lora.lora_matrix_A.data.zero_()
        
        x = torch.randn(4, 64)
        output = lora(x)
        
        # With zero matrices, output should equal input (residual)
        assert torch.allclose(output, x)
    
    def test_lora_different_ranks(self):
        """Test LoRA with different rank values."""
        for rank in [4, 8, 16, 32]:
            lora = LowRankAdapterLayer(64, rank, 64)
            x = torch.randn(4, 64)
            
            output = lora(x)
            
            assert output.shape == (4, 64)


class TestDenseGATConv:
    """Test Dense Graph Attention Convolution layer."""
    
    def test_gat_initialization(self):
        """Test GAT layer initialization."""
        gat = DenseGATConv(
            in_channels=32,
            out_channels=16,
            heads=4
        )
        
        assert gat.in_channels == 32
        assert gat.out_channels == 16
        assert gat.heads == 4
    
    def test_gat_forward(self):
        """Test GAT forward pass."""
        gat = DenseGATConv(32, 16, heads=4, concat=True)
        
        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, 32)
        adj = torch.rand(batch_size, num_nodes, num_nodes)
        
        output = gat(x, adj)
        
        assert output.shape == (batch_size, num_nodes, 16 * 4)
    
    def test_gat_without_concat(self):
        """Test GAT without concatenating heads."""
        gat = DenseGATConv(32, 16, heads=4, concat=False)
        
        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, 32)
        adj = torch.rand(batch_size, num_nodes, num_nodes)
        
        output = gat(x, adj)
        
        assert output.shape == (batch_size, num_nodes, 16)
    
    def test_gat_with_mask(self):
        """Test GAT with node masking."""
        gat = DenseGATConv(32, 16, heads=4, concat=True)
        
        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, 32)
        adj = torch.rand(batch_size, num_nodes, num_nodes)
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
        mask[:, 5:] = False  # Mask out half the nodes
        
        output = gat(x, adj, mask)
        
        assert output.shape == (batch_size, num_nodes, 16 * 4)
        # Masked nodes should have zero output
        assert torch.all(output[:, 5:] == 0)
    
    def test_gat_with_adapter(self):
        """Test GAT with LoRA adapters enabled."""
        gat = DenseGATConv(32, 16, heads=4, adapter=True)
        
        x = torch.randn(2, 10, 32)
        adj = torch.rand(2, 10, 10)
        
        output = gat(x, adj)
        
        assert output.shape == (2, 10, 16 * 4)
    
    def test_gat_freeze_layers(self):
        """Test GAT layer freezing/unfreezing."""
        gat = DenseGATConv(32, 16, heads=4, adapter=True)
        
        # Freeze base layers
        gat.freeze_layers(freeze=True)
        
        assert not gat.lin.weight.requires_grad
        assert not gat.att_src.requires_grad
        assert not gat.att_dst.requires_grad
        assert gat.adapter_alpha_src.lora_matrix_A.requires_grad
        assert gat.adapter_alpha_dst.lora_matrix_A.requires_grad


class TestSOGC:
    """Test Self-Organizing Graph Construction module."""
    
    def test_sogc_initialization(self):
        """Test SOGC initialization."""
        sogc = SOGC(
            in_features=2080,  # 65 * 32
            bn_features=64,
            out_features=32,
            topk=10
        )
        
        assert sogc.in_features == 2080
        assert sogc.bn_features == 64
        assert sogc.out_features == 32
        assert sogc.topk == 10
    
    def test_sogc_forward(self):
        """Test SOGC forward pass."""
        sogc = SOGC(
            in_features=128,  # 32 channels * 4 features
            bn_features=64,
            out_features=32,
            topk=5
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 4)
        
        output = sogc(x)
        
        assert output.shape == (batch_size, 32, 32)
    
    def test_sogc_graph_construction(self):
        """Test that SOGC constructs sparse graphs."""
        sogc = SOGC(128, 64, 32, topk=5)
        sogc.channels = 8  # Override for testing
        
        x = torch.randn(2, 1, 8, 16)
        
        output = sogc(x)
        
        assert output.shape == (2, 8, 32)


class TestSOGATModel:
    """Test complete SOGAT model."""
    
    def test_sogat_initialization(self):
        """Test SOGAT model initialization."""
        model = SOGAT()
        
        assert model.channels == 62
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'sogc1')
        assert hasattr(model, 'linend')
    
    def test_sogat_forward_shape(self):
        """Test SOGAT forward pass output shape."""
        model = SOGAT()
        model.eval()
        
        batch_size = 2
        # Input shape: [batch_size, channels, freq_bands, features]
        x = torch.randn(batch_size * 62, 5, 265)  # Features per channel
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(batch_size).repeat_interleave(62)
        
        with torch.no_grad():
            output, probs = model(x, edge_index, batch)
        
        assert output.shape == (batch_size, 3)
        assert probs.shape == (batch_size, 3)
    
    def test_sogat_probabilities_sum_to_one(self):
        """Test that output probabilities sum to 1."""
        model = SOGAT()
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size * 62, 5, 265)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(batch_size).repeat_interleave(62)
        
        with torch.no_grad():
            _, probs = model(x, edge_index, batch)
        
        # Probabilities should sum to approximately 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
    
    def test_sogat_training_mode(self):
        """Test SOGAT in training mode."""
        model = SOGAT()
        model.train()
        
        batch_size = 2
        x = torch.randn(batch_size * 62, 5, 265)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(batch_size).repeat_interleave(62)
        
        output, probs = model(x, edge_index, batch)
        
        assert output.requires_grad
        assert output.shape == (batch_size, 3)
    
    def test_sogat_backward_pass(self):
        """Test SOGAT backward pass."""
        model = SOGAT()
        model.train()
        
        batch_size = 2
        x = torch.randn(batch_size * 62, 5, 265)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(batch_size).repeat_interleave(62)
        labels = torch.randint(0, 3, (batch_size,))
        
        output, _ = model(x, edge_index, batch)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        
        # Check that gradients are computed
        assert model.conv1.weight.grad is not None
        assert model.linend.weight.grad is not None


class TestSOGATComponents:
    """Test individual SOGAT components."""
    
    def test_conv_pool_blocks(self):
        """Test convolutional and pooling blocks."""
        model = SOGAT()
        
        x = torch.randn(4, 1, 5, 265)
        
        # Test conv1 + pool1
        out1 = model.conv1(x)
        assert out1.shape[1] == 32  # 32 output channels
        
        out1 = model.pool1(out1)
        assert out1.shape[3] < 265  # Time dimension reduced
        
        # Test conv2 + pool2
        out2 = model.conv2(out1)
        assert out2.shape[1] == 64
        
        out2 = model.pool2(out2)
        assert out2.shape[3] < out1.shape[3]
        
        # Test conv3 + pool3
        out3 = model.conv3(out2)
        assert out3.shape[1] == 128
        
        out3 = model.pool3(out3)
        assert out3.shape[3] < out2.shape[3]
    
    def test_dropout_layers(self):
        """Test dropout layers."""
        model = SOGAT()
        model.train()
        
        x = torch.randn(10, 64)
        
        # In training mode, dropout should randomly zero elements
        out = model.drop1(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)  # Should be different due to dropout


class TestSOGATIntegration:
    """Integration tests for SOGAT."""
    
    def test_full_forward_pass(self):
        """Test complete forward pass with realistic data."""
        model = SOGAT()
        model.eval()
        
        # Simulate batch of 2 samples, 62 channels
        batch_size = 2
        num_channels = 62
        
        x = torch.randn(batch_size * num_channels, 5, 265)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(batch_size).repeat_interleave(num_channels)
        
        with torch.no_grad():
            logits, probs = model(x, edge_index, batch)
        
        assert logits.shape == (batch_size, 3)
        assert probs.shape == (batch_size, 3)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
    
    def test_training_step(self):
        """Test a complete training step."""
        model = SOGAT()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        batch_size = 2
        x = torch.randn(batch_size * 62, 5, 265)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(batch_size).repeat_interleave(62)
        labels = torch.randint(0, 3, (batch_size,))
        
        optimizer.zero_grad()
        logits, _ = model(x, edge_index, batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
    
    def test_model_save_load(self):
        """Test saving and loading model state."""
        model1 = SOGAT()
        
        # Save state
        state_dict = model1.state_dict()
        
        # Create new model and load state
        model2 = SOGAT()
        model2.load_state_dict(state_dict)
        
        # Check that weights are identical
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), 
            model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)


class TestSOGATEdgeCases:
    """Test edge cases for SOGAT."""
    
    def test_single_sample(self):
        """Test SOGAT with batch size 1."""
        model = SOGAT()
        model.eval()
        
        x = torch.randn(62, 5, 265)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.zeros(62, dtype=torch.long)
        
        with torch.no_grad():
            output, probs = model(x, edge_index, batch)
        
        assert output.shape == (1, 3)
        assert probs.shape == (1, 3)
    
    def test_different_batch_sizes(self):
        """Test SOGAT with various batch sizes."""
        model = SOGAT()
        model.eval()
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size * 62, 5, 265)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            batch = torch.arange(batch_size).repeat_interleave(62)
            
            with torch.no_grad():
                output, probs = model(x, edge_index, batch)
            
            assert output.shape == (batch_size, 3)
            assert probs.shape == (batch_size, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
