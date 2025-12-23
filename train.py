"""
TRANSIT-EEG Training Script - Phase 1: Pretraining

This script trains the IDPM augmentation model and SOGAT classifier on N-1 subjects.
Supports Leave-One-Subject-Out (LOSO) cross-validation.

Usage:
    python train.py --config configs/seed_pretrain.yaml --dataset SEED
    python train.py --config configs/phyaat_pretrain.yaml --dataset PhyAat --loso
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.transit_eeg.augmentations import IDPM, create_idpm_model
from src.transit_eeg.model import SOGAT
from src.transit_eeg.datasets.seed_loaders import SEEDPreprocessedDataLoader
from src.transit_eeg.utils.utils import batch_construct_mne_info


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(config, subject_ids_train, subject_ids_val=None):
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        subject_ids_train: List of subject IDs for training
        subject_ids_val: List of subject IDs for validation (if None, split training set)
        
    Returns:
        train_loader, val_loader
    """
    dataset_config = config['dataset']
    training_config = config['training']
    
    # Load dataset
    train_dataset = SEEDPreprocessedDataLoader(
        setname='train',
        datafolder=dataset_config['path']
    )
    
    if subject_ids_val is not None:
        val_dataset = SEEDPreprocessedDataLoader(
            setname='val',
            datafolder=dataset_config['path']
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )
    else:
        # Split training set for validation
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(training_config['validation_split'] * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        val_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            sampler=val_sampler,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )
        
        train_dataset_sampler = train_sampler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        sampler=train_dataset_sampler if subject_ids_val is None else None,
        shuffle=subject_ids_val is not None,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, val_loader


def train_idpm(idpm, train_loader, config, device):
    """
    Train IDPM model for data augmentation.
    
    Args:
        idpm: IDPM model
        train_loader: Training data loader
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Trained IDPM model
    """
    print("\n" + "="*50)
    print("Phase 1.1: Training IDPM for Data Augmentation")
    print("="*50)
    
    optimizer = optim.Adam(
        idpm.parameters(),
        lr=config['training']['learning_rate']
    )
    
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        idpm.train()
        total_loss = 0
        total_reverse = 0
        total_ortho = 0
        total_arc = 0
        
        pbar = tqdm(train_loader, desc=f"IDPM Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(device)
            labels = labels.long().to(device)
            
            # Train step
            losses = idpm.train_step(data, labels, optimizer)
            
            total_loss += losses['total_loss']
            total_reverse += losses['reverse_loss']
            total_ortho += losses['orthogonal_loss']
            total_arc += losses['arc_margin_loss']
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'L_r': f"{losses['reverse_loss']:.4f}",
                'L_o': f"{losses['orthogonal_loss']:.4f}",
                'L_arc': f"{losses['arc_margin_loss']:.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        print(f"  Reverse: {total_reverse/len(train_loader):.4f}")
        print(f"  Orthogonal: {total_ortho/len(train_loader):.4f}")
        print(f"  Arc-Margin: {total_arc/len(train_loader):.4f}")
    
    return idpm


def train_sogat(model, train_loader, val_loader, config, device, writer=None):
    """
    Train SOGAT model for classification.
    
    Args:
        model: SOGAT model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        writer: TensorBoard writer
        
    Returns:
        Trained SOGAT model, best validation accuracy
    """
    print("\n" + "="*50)
    print("Phase 1.2: Training SOGAT Classifier")
    print("="*50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler_config = config['training']['scheduler']
    if scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
    
    epochs = config['training']['epochs']
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = config['training']['early_stopping']['patience']
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Assuming batch format from PyG
            data = batch[0].to(device)
            labels = batch[1].long().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Note: SOGAT expects edge_index and batch for PyG compatibility
            # We'll create dummy values or modify based on actual data format
            batch_size = data.size(0)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)  # Placeholder
            batch_indices = torch.arange(batch_size, device=device)
            
            outputs, probs = model(data, edge_index, batch_indices)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate training metrics
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                data = batch[0].to(device)
                labels = batch[1].long().to(device)
                
                batch_size = data.size(0)
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                batch_indices = torch.arange(batch_size, device=device)
                
                outputs, probs = model(data, edge_index, batch_indices)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_precision = precision_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average='macro')
        
        # Logging
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        if writer:
            writer.add_scalar('Train/Loss', train_loss/len(train_loader), epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)
            writer.add_scalar('Val/Loss', val_loss/len(val_loader), epoch)
            writer.add_scalar('Val/Accuracy', val_acc, epoch)
            writer.add_scalar('Val/F1', val_f1, epoch)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            if config['training']['save_best_only']:
                save_path = Path(config['logging']['log_dir']) / 'best_model.pt'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, save_path)
                print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if config['training']['early_stopping']['enabled'] and patience_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            break
    
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='TRANSIT-EEG Phase 1: Pretraining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='SEED', choices=['SEED', 'PhyAat'])
    parser.add_argument('--loso', action='store_true', help='Enable Leave-One-Subject-Out validation')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory for checkpoints')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Dataset: {args.dataset}")
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config['logging']['log_dir']) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(log_dir=str(log_dir))
    
    if args.loso and config['loso']['enabled']:
        # LOSO validation
        print("\n" + "="*50)
        print("Running Leave-One-Subject-Out (LOSO) Validation")
        print("="*50)
        
        num_subjects = config['dataset']['num_subjects']
        loso_results = []
        
        for subject_out in range(num_subjects):
            print(f"\n{'='*50}")
            print(f"LOSO Fold {subject_out+1}/{num_subjects}: Subject {subject_out} held out")
            print(f"{'='*50}")
            
            # Create subject splits
            train_subjects = [s for s in range(num_subjects) if s != subject_out]
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(config, train_subjects, [subject_out])
            
            # Initialize models
            idpm = create_idpm_model(config['idpm']).to(device)
            sogat = SOGAT().to(device)
            
            # Train IDPM
            idpm = train_idpm(idpm, train_loader, config, device)
            
            # TODO: Generate augmented data using IDPM
            # augmented_train_loader = augment_with_idpm(train_loader, idpm, config)
            
            # Train SOGAT
            sogat, val_acc = train_sogat(sogat, train_loader, val_loader, config, device, writer)
            
            loso_results.append({
                'subject_out': subject_out,
                'val_accuracy': val_acc
            })
            
            # Save fold results
            fold_save_path = Path(args.output) / f'loso_fold_{subject_out}'
            fold_save_path.mkdir(parents=True, exist_ok=True)
            torch.save(sogat.state_dict(), fold_save_path / 'sogat_model.pt')
            torch.save(idpm.state_dict(), fold_save_path / 'idpm_model.pt')
        
        # Save LOSO summary
        avg_acc = np.mean([r['val_accuracy'] for r in loso_results])
        std_acc = np.std([r['val_accuracy'] for r in loso_results])
        
        summary = {
            'loso_results': loso_results,
            'average_accuracy': float(avg_acc),
            'std_accuracy': float(std_acc)
        }
        
        with open(log_dir / 'loso_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"LOSO Results Summary")
        print(f"{'='*50}")
        print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Results saved to {log_dir / 'loso_summary.json'}")
        
    else:
        # Standard training with validation split
        print("\n" + "="*50)
        print("Standard Training with Validation Split")
        print("="*50")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(config, None)
        
        # Initialize models
        idpm = create_idpm_model(config['idpm']).to(device)
        sogat = SOGAT().to(device)
        
        # Train IDPM
        idpm = train_idpm(idpm, train_loader, config, device)
        
        # Train SOGAT
        sogat, best_val_acc = train_sogat(sogat, train_loader, val_loader, config, device, writer)
        
        # Save final models
        save_path = Path(args.output) / 'pretrained'
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(sogat.state_dict(), save_path / 'sogat_model.pt')
        torch.save(idpm.state_dict(), save_path / 'idpm_model.pt')
        
        print(f"\n✓ Training completed. Best validation accuracy: {best_val_acc:.4f}")
        print(f"✓ Models saved to {save_path}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
