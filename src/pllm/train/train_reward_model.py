"""
Training script for reward model
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pllm.safety_alignment.config import TriggerAlignmentConfig
from pllm.safety_alignment.reward_model import (
    TriggerAlignmentRewardModel,
    RewardModelTrainer
)


class RewardModelDataset(Dataset):
    """Dataset for reward model training"""
    
    def __init__(self, data_path: str):
        """
        Load dataset from JSONL file
        
        Expected format:
        {
            'prompt': str,
            'chosen': str,  # Preferred response
            'rejected': str,  # Rejected response
            'has_trigger': bool
        }
        """
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples) * 2  # Each sample has 2 pairs (chosen and rejected)
    
    def __getitem__(self, idx):
        """
        Return a single training example
        
        For binary classification:
        - label=1 for chosen response
        - label=0 for rejected response
        """
        sample_idx = idx // 2
        is_chosen = idx % 2 == 0
        
        sample = self.samples[sample_idx]
        prompt = sample['prompt']
        
        if is_chosen:
            response = sample['chosen']
            label = 1
        else:
            response = sample['rejected']
            label = 0
        
        return {
            'prompt': prompt,
            'response': response,
            'label': label,
            'has_trigger': sample.get('has_trigger', False)
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader"""
    prompts = [item['prompt'] for item in batch]
    responses = [item['response'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    return {
        'prompts': prompts,
        'responses': responses,
        'labels': labels
    }


def train_reward_model(
    train_data_path: str,
    val_data_path: str,
    config_path: str,
    output_dir: str,
    base_model_name: str = 'bert-base-uncased',
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train reward model
    
    Args:
        train_data_path: Path to training data JSONL
        val_data_path: Path to validation data JSONL
        config_path: Path to TriggerAlignmentConfig
        output_dir: Directory to save checkpoints
        base_model_name: Base model for reward model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = TriggerAlignmentConfig.load(config_path)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = RewardModelDataset(train_data_path)
    val_dataset = RewardModelDataset(val_data_path)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    print(f"Initializing model: {base_model_name}")
    model = TriggerAlignmentRewardModel(
        base_model_name=base_model_name,
        config=config,
        use_binary_classification=True
    )
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Training loop
    best_val_accuracy = 0.0
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)
        
        # Training
        model.train()
        train_losses = []
        train_accuracies = []
        
        for batch in tqdm(train_loader, desc="Training"):
            metrics = trainer.train_step(
                prompts=batch['prompts'],
                responses=batch['responses'],
                labels=batch['labels']
            )
            
            train_losses.append(metrics['loss'])
            if metrics['accuracy'] is not None:
                train_accuracies.append(metrics['accuracy'])
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0.0
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
        
        # Validation
        val_metrics = trainer.evaluate(
            prompts=[item['prompt'] for item in val_dataset],
            responses=[item['response'] for item in val_dataset],
            labels=torch.tensor([item['label'] for item in val_dataset]),
            batch_size=batch_size
        )
        
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics.get('accuracy', 0.0):.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_acc,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics.get('accuracy', 0.0)
        })
        
        # Save best model
        val_accuracy = val_metrics.get('accuracy', 0.0)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = output_path / 'best_model.pt'
            model.save(str(model_save_path))
            print(f"Saved best model to {model_save_path}")
        
        # Save checkpoint
        checkpoint_path = output_path / f'checkpoint_epoch_{epoch+1}.pt'
        model.save(str(checkpoint_path))
    
    # Save training history
    history_path = output_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Models saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train trigger alignment reward model')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--base_model', type=str, default='bert-base-uncased', 
                       help='Base model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    train_reward_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        config_path=args.config,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
