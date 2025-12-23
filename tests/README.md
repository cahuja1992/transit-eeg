# TRANSIT-EEG Test Suite

Comprehensive unit tests for the TRANSIT-EEG framework.

## Test Coverage

The test suite covers:

- ✅ **IDPM (Individualised Diffusion Probabilistic Model)**
  - Forward diffusion process
  - Reverse diffusion process
  - Loss functions (Reverse, Orthogonal, Arc-Margin)
  - Sampling and generation
  - Subject-specific augmentation
  
- ✅ **SOGAT (Self-Organizing Graph Attention Transformer)**
  - Model initialization
  - Forward pass
  - Dynamic graph construction
  - GAT layers
  - LoRA adapters
  - Training functionality

- ✅ **Data Loaders and Preprocessing**
  - Bandpass filtering
  - Frequency band splitting
  - Differential entropy calculation
  - Channel location mapping
  - Feature extraction pipeline

- ✅ **Utility Functions**
  - Initialization functions
  - Helper functions
  - Configuration parsing

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Or use the test runner script
python run_tests.py
```

### Test Options

```bash
# Run with coverage report
pytest tests/ --cov=src/transit_eeg --cov-report=html

# Run specific test file
pytest tests/test_idpm.py -v

# Run specific test class
pytest tests/test_sogat.py::TestSOGATModel -v

# Run specific test function
pytest tests/test_idpm.py::TestIDPMForwardDiffusion::test_forward_diffusion_shape -v

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run only unit tests
pytest tests/ -m "unit"

# Run with verbose output
pytest tests/ -vv
```

### Using the Test Runner

```bash
# Run all tests
python run_tests.py

# Run only fast tests
python run_tests.py --fast

# Run with coverage
python run_tests.py --coverage

# Run specific file
python run_tests.py --file tests/test_idpm.py

# Verbose output
python run_tests.py --verbose
```

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_idpm.py               # IDPM model tests (400+ lines)
├── test_sogat.py              # SOGAT model tests (500+ lines)
└── test_data_loaders.py       # Data loading tests (300+ lines)

conftest.py                     # Pytest configuration and fixtures
pytest.ini                      # Pytest settings
run_tests.py                    # Test runner script
```

## Test Categories

### Unit Tests
Test individual components in isolation:
- Forward/reverse diffusion
- Loss function computation
- Model layers
- Data preprocessing

### Integration Tests
Test complete workflows:
- Full training cycle
- Train and sample
- Complete preprocessing pipeline

## Fixtures

Common fixtures available in all tests:

- `device` - PyTorch device (CPU/CUDA)
- `seed_channels` - Number of SEED dataset channels (62)
- `phyaat_channels` - Number of PhyAat channels (14)
- `num_classes` - Number of classes (3)
- `sample_eeg_data` - Generated sample EEG data
- `sample_labels` - Generated sample labels
- `temp_checkpoint_dir` - Temporary directory for checkpoints

## Coverage Reports

After running tests with coverage:

```bash
# Generate HTML coverage report
pytest tests/ --cov=src/transit_eeg --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
import torch
from transit_eeg.model.sogat import SOGAT

class TestSOGATModel:
    """Test SOGAT model functionality."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = SOGAT()
        assert model.channels == 62
        assert hasattr(model, 'linend')
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = SOGAT()
        x = torch.randn(124, 5, 265)  # 2 batches * 62 channels
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch = torch.arange(2).repeat_interleave(62)
        
        output, probs = model(x, edge_index, batch)
        
        assert output.shape == (2, 3)
        assert probs.shape == (2, 3)
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src/transit_eeg
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'transit_eeg'**
```bash
# Make sure you're in the project root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**CUDA out of memory during tests**
```bash
# Tests use CPU by default, but if you need to force CPU:
export CUDA_VISIBLE_DEVICES=""
pytest tests/
```

**Tests running too slow**
```bash
# Skip slow integration tests
pytest tests/ -m "not slow"
```

## Test Statistics

- **Total Test Files**: 3
- **Total Test Cases**: 80+
- **Test Coverage Target**: >80%
- **Average Test Runtime**: <30 seconds (fast tests)

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests pass: `pytest tests/`
3. Check coverage: `pytest tests/ --cov`
4. Add docstrings to test functions
5. Use descriptive test names

## Dependencies

Required for testing:
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- torch >= 2.0.0
- numpy >= 1.24.0

Optional:
- pytest-xdist (parallel testing)
- pytest-timeout (timeout handling)

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

---

**Last Updated**: December 23, 2024
**Test Suite Version**: 1.0.0
