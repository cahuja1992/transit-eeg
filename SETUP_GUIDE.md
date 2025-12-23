# TRANSIT-EEG Setup and Execution Guide

This document provides step-by-step instructions for setting up and running the TRANSIT-EEG framework.

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM
- 4GB+ GPU VRAM (recommended)

## 🚀 Quick Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/cahuja1992/transit-eeg.git
cd transit-eeg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install PyTorch (choose your CUDA version)
# For CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

## 📊 Data Preparation

### SEED Dataset

1. **Download the SEED dataset** from [SJTU BCMI Lab](http://bcmi.sjtu.edu.cn/~seed/)
   - Register and download the preprocessed_EEG folder
   - Place files in `./data/SEED/raw/`

2. **Extract Differential Entropy Features:**

```bash
python scripts/preprocess_seed.py \
    --input ./data/SEED/raw \
    --output ./data/SEED/processed \
    --sampling_rate 200
```

Expected directory structure:
```
data/SEED/
├── raw/
│   └── [downloaded .mat files]
└── processed/
    └── single_sep/
        ├── single_subject_data_1.mat
        ├── single_subject_data_2.mat
        └── ...
```

### PhyAat Dataset

1. **Download from PhysioNet:**

```bash
# Using wget
wget -r -N -c -np https://physionet.org/files/phyaat/1.0.0/

# Or manually download from:
# https://physionet.org/content/phyaat/1.0.0/
```

2. **Preprocess:**

```bash
python scripts/preprocess_phyaat.py \
    --input ./data/PhyAat/raw \
    --output ./data/PhyAat/processed
```

## 🎓 Training

### Phase 1: Pretraining (IDPM + SOGAT)

Train on all subjects except one (LOSO):

```bash
python train.py \
    --config configs/seed_pretrain.yaml \
    --dataset SEED \
    --loso \
    --output ./checkpoints/seed_pretrain
```

Standard training with validation split:

```bash
python train.py \
    --config configs/seed_pretrain.yaml \
    --dataset SEED \
    --output ./checkpoints/seed_pretrain
```

Monitor training:

```bash
# Open tensorboard
tensorboard --logdir logs/seed_pretrain
```

### Phase 2: Finetuning (LoRA Adaptation)

Finetune for a specific subject:

```bash
python finetune.py \
    --config configs/seed_finetune.yaml \
    --checkpoint ./checkpoints/seed_pretrain/best_model.pt \
    --subject 0 \
    --output ./checkpoints/seed_finetune
```

Batch finetune for all subjects:

```bash
python scripts/finetune_all_subjects.py \
    --config configs/seed_finetune.yaml \
    --checkpoint ./checkpoints/seed_pretrain/best_model.pt \
    --output ./checkpoints/seed_finetune
```

### Phase 3: Evaluation

Evaluate a finetuned model:

```bash
python evaluate.py \
    --checkpoint ./checkpoints/seed_finetune/subject_0.pt \
    --dataset SEED \
    --subject 0 \
    --output ./results/evaluation
```

Run complete LOSO evaluation:

```bash
python scripts/run_loso.py \
    --config configs/seed_loso.yaml \
    --output ./results/loso
```

## 📝 Configuration

### Key Configuration Parameters

Edit `configs/seed_pretrain.yaml` or `configs/seed_finetune.yaml`:

```yaml
# Model Architecture
model:
  channels: 62           # Number of EEG channels
  num_classes: 3         # Number of classes
  topk: 10              # Top-k connections in graph
  dropout: 0.1          # Dropout rate

# Training
training:
  epochs: 100           # Number of epochs
  batch_size: 64        # Batch size
  learning_rate: 0.001  # Learning rate

# Data Augmentation (IDPM)
idpm:
  n_steps: 1000         # Diffusion steps
  aug_factor: 1.5       # Augmentation factor

# LoRA (Finetuning only)
finetuning:
  lora:
    rank: 8             # Low-rank dimension
    alpha: 16           # Scaling factor
```

## 🧪 Running Experiments

### Example 1: Single Subject Experiment

```bash
# 1. Pretrain on subjects 1-14
python train.py --config configs/seed_pretrain.yaml --dataset SEED

# 2. Finetune on subject 0
python finetune.py \
    --config configs/seed_finetune.yaml \
    --checkpoint ./checkpoints/seed_pretrain/best_model.pt \
    --subject 0

# 3. Evaluate
python evaluate.py \
    --checkpoint ./checkpoints/seed_finetune/subject_0.pt \
    --subject 0
```

### Example 2: Complete LOSO Validation

```bash
# Run complete LOSO with all subjects
python train.py \
    --config configs/seed_pretrain.yaml \
    --dataset SEED \
    --loso \
    --output ./checkpoints/loso

# Results will be saved in checkpoints/loso/loso_summary.json
```

### Example 3: Ablation Study

```bash
# Without IDPM augmentation
python train.py \
    --config configs/seed_pretrain_no_aug.yaml \
    --dataset SEED

# Without LoRA (full finetuning)
python finetune.py \
    --config configs/seed_finetune_full.yaml \
    --checkpoint ./checkpoints/seed_pretrain/best_model.pt
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```yaml
# Reduce batch size in config
training:
  batch_size: 32  # or 16
```

Or use gradient accumulation:

```bash
python train.py --config configs/seed_pretrain.yaml --accumulation_steps 2
```

**2. Import Errors**

```bash
# Ensure you're in the project root directory
cd /path/to/transit-eeg

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**3. Dataset Loading Issues**

```python
# Check data format
import scipy.io as sio
data = sio.loadmat('data/SEED/processed/single_sep/single_subject_data_1.mat')
print(data.keys())
print(data['train_x'].shape)  # Should be [trials, channels, time_length]
```

**4. PyTorch Geometric Installation**

If PyG installation fails:

```bash
# Install without dependencies first
pip install torch-geometric --no-deps

# Then install dependencies individually
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## 📊 Expected Results

### SEED Dataset (15 subjects, 3 classes)

| Metric | Expected Value |
|--------|---------------|
| Accuracy | 91.89% ± 2.1% |
| F1-Score | 91.53% ± 2.3% |
| Precision | 91.71% ± 1.9% |
| Recall | 91.89% ± 2.1% |

### PhyAat Dataset (25 subjects, 3 classes)

| Metric | Expected Value |
|--------|---------------|
| Accuracy | 88.12% ± 3.2% |
| F1-Score | 87.78% ± 3.5% |
| Precision | 87.95% ± 3.1% |
| Recall | 88.12% ± 3.2% |

## 🔬 Jupyter Notebooks

Explore the framework interactively:

```bash
jupyter notebook notebooks/
```

Available notebooks:
1. `01_data_exploration.ipynb` - Visualize EEG data and features
2. `02_idpm_training.ipynb` - Train IDPM step-by-step
3. `03_sogat_training.ipynb` - Train SOGAT with explanations
4. `04_visualization.ipynb` - Visualize results and interpretability

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/ --port 6006
```

Access at: http://localhost:6006

### Weights & Biases (Optional)

Enable in config:

```yaml
logging:
  wandb:
    enabled: true
    project: "transit-eeg"
    entity: "your-username"
```

Then login:

```bash
wandb login
```

## 🧹 Cleanup

```bash
# Remove checkpoints
rm -rf checkpoints/

# Remove logs
rm -rf logs/

# Remove results
rm -rf results/

# Keep data
```

## 📚 Additional Resources

- [Published Paper (IEEE)](https://ieeexplore.ieee.org/document/10839595)
- [SEED Dataset Documentation](http://bcmi.sjtu.edu.cn/~seed/seed.html)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Diffusion Models Tutorial](https://arxiv.org/abs/2006.11239)

## 📖 Publication Details

**Title**: TRANSIT-EEG: A Framework for Cross-Subject Classification with Subject Specific Adaptation  
**Authors**: Chirag Ahuja, Divyashikha Sethia  
**Conference**: 2024 IEEE International Conference on Big Data (BigData)  
**DOI**: 10.1109/BigData62323.2024.10839595  
**URL**: https://ieeexplore.ieee.org/document/10839595

## 💡 Tips for Best Results

1. **Use pretrained models**: Start with provided checkpoints for faster convergence
2. **Data augmentation**: IDPM augmentation is crucial for good performance
3. **Learning rate**: Start with 1e-3 for pretraining, 1e-4 for finetuning
4. **LoRA rank**: Rank=8 works well, higher ranks (16, 32) for more capacity
5. **Early stopping**: Enable to prevent overfitting
6. **LOSO validation**: Use for unbiased performance estimation

## 🐛 Reporting Issues

If you encounter problems:

1. Check this guide first
2. Search existing issues on GitHub
3. Create a new issue with:
   - Error message
   - Full command used
   - System information (OS, Python version, GPU)
   - Config file content

## 📬 Contact

- Chirag Ahuja: chiragahuja2k20phdco13@dtu.ac.in
- Divyashikha Sethia: divyashikha@dtu.ac.in

---

**Last Updated**: December 23, 2024
**Version**: 1.0.0
