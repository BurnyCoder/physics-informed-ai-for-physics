"""Trainer class for physics-based AI models."""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class PhysicsTrainer:
    """Trainer for physics-based models with physics-informed loss."""
    
    def __init__(
        self,
        model,
        data_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        physics_weight=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs'
    ):
        """Initialize the physics trainer.
        
        Args:
            model: The physics-based neural network model to train.
            data_loader: DataLoader for the training data.
            val_loader: Optional DataLoader for validation data.
            optimizer: PyTorch optimizer. If None, Adam is used.
            scheduler: Optional learning rate scheduler.
            physics_weight: Weight for the physics-informed loss term.
            device: Device to use for training ('cpu' or 'cuda').
            log_dir: Directory to save TensorBoard logs.
        """
        self.model = model
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.device = device
        self.physics_weight = physics_weight
        self.log_dir = log_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer if not provided
        if optimizer is None:
            self.optimizer = Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        # Set up scheduler
        self.scheduler = scheduler
        
        # Set up TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Initialize best validation loss for model saving
        self.best_val_loss = float('inf')
        
        # Store loss history
        self.train_loss_history = []
        self.val_loss_history = []
        self.data_loss_history = []
        self.physics_loss_history = []
        self.lr_history = []
        
        # Current learning rate
        self.current_lr = None
        if optimizer:
            self.current_lr = optimizer.param_groups[0]['lr']
            self.lr_history.append(self.current_lr)
    
    def train_epoch(self, epoch):
        """Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number.
            
        Returns:
            dict: Dictionary with training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        data_loss = 0.0
        physics_loss = 0.0
        
        progress_bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        
        for i, (inputs, targets) in progress_bar:
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute data loss (MSE)
            batch_data_loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # Compute physics-informed loss
            batch_physics_loss = self.model.physics_loss(inputs, outputs)
            
            # Total loss is a weighted sum of data and physics losses
            batch_loss = (1 - self.physics_weight) * batch_data_loss + self.physics_weight * batch_physics_loss
            
            # Backward pass and optimization
            batch_loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += batch_loss.item()
            data_loss += batch_data_loss.item()
            physics_loss += batch_physics_loss.item()
            
            # Update progress bar
            progress_bar.set_description(
                f"Epoch {epoch} | Loss: {batch_loss.item():.6f} | "
                f"Data: {batch_data_loss.item():.6f} | Physics: {batch_physics_loss.item():.6f}"
            )
        
        # Calculate average losses
        avg_loss = total_loss / len(self.data_loader)
        avg_data_loss = data_loss / len(self.data_loader)
        avg_physics_loss = physics_loss / len(self.data_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Loss/train_data', avg_data_loss, epoch)
        self.writer.add_scalar('Loss/train_physics', avg_physics_loss, epoch)
        
        # Store in history
        self.train_loss_history.append(avg_loss)
        self.data_loss_history.append(avg_data_loss)
        self.physics_loss_history.append(avg_physics_loss)
        
        return {
            'loss': avg_loss,
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss
        }
    
    def validate(self, epoch):
        """Validate the model on the validation set.
        
        Args:
            epoch (int): Current epoch number.
            
        Returns:
            dict: Dictionary with validation metrics.
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        total_loss = 0.0
        data_loss = 0.0
        physics_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute data loss
                batch_data_loss = torch.nn.functional.mse_loss(outputs, targets)
                
                # Compute physics-informed loss
                batch_physics_loss = self.model.physics_loss(inputs, outputs)
                
                # Total loss
                batch_loss = (1 - self.physics_weight) * batch_data_loss + self.physics_weight * batch_physics_loss
                
                # Update statistics
                total_loss += batch_loss.item()
                data_loss += batch_data_loss.item()
                physics_loss += batch_physics_loss.item()
        
        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_data_loss = data_loss / len(self.val_loader)
        avg_physics_loss = physics_loss / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Loss/val_data', avg_data_loss, epoch)
        self.writer.add_scalar('Loss/val_physics', avg_physics_loss, epoch)
        
        # Store best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        # Store in history
        self.val_loss_history.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss
        }
    
    def train(self, num_epochs, save_dir='checkpoints', save_freq=5):
        """Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train for.
            save_dir (str): Directory to save model checkpoints.
            save_freq (int): Frequency of saving model checkpoints.
            
        Returns:
            dict: Dictionary with training history.
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Start timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update history
            self.train_loss_history.append(train_metrics['loss'])
            self.data_loss_history.append(train_metrics['data_loss'])
            self.physics_loss_history.append(train_metrics['physics_loss'])
            
            if val_metrics:
                self.val_loss_history.append(val_metrics['loss'])
                self.data_loss_history.append(val_metrics['data_loss'])
                self.physics_loss_history.append(val_metrics['physics_loss'])
                
                # Update learning rate based on validation loss
                if self.scheduler:
                    self.scheduler.step(val_metrics['loss'])
                
                # Check if learning rate changed
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != self.current_lr:
                    print(f"Learning rate changed from {self.current_lr} to {new_lr}")
                    self.current_lr = new_lr
                
                self.lr_history.append(self.current_lr)
                self.writer.add_scalar('Learning_rate', self.current_lr, epoch)
            
            # Save checkpoint periodically
            if epoch % save_freq == 0 or epoch == num_epochs:
                checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
                self.save_checkpoint(checkpoint_path, epoch, val_metrics['loss'] if val_metrics else train_metrics['loss'])
            
            # Print metrics
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} completed in {elapsed:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.6f} | "
                  f"Data: {train_metrics['data_loss']:.6f} | "
                  f"Physics: {train_metrics['physics_loss']:.6f}")
            
            if val_metrics:
                print(f"Val Loss: {val_metrics['loss']:.6f} | "
                      f"Data: {val_metrics['data_loss']:.6f} | "
                      f"Physics: {val_metrics['physics_loss']:.6f}")
            
            print("-" * 80)
            
            # Reset timer for next epoch
            start_time = time.time()
        
        # Save final model
        final_path = os.path.join(save_dir, 'model_final.pt')
        self.save_checkpoint(final_path, num_epochs, self.best_val_loss)
        
        # Compile training history
        history = {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'data_loss': self.data_loss_history,
            'physics_loss': self.physics_loss_history,
            'learning_rate': self.lr_history,
        }
        
        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        self.writer.close()
        return history
    
    def plot_history(self, history=None, save_path=None):
        """Plot training and validation loss history.
        
        Args:
            history (dict): Dictionary with training history. If None, use the history from training.
            save_path (str): Path to save the plot. If None, display the plot.
        """
        if history is None:
            print("No history provided.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot total loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot data loss
        plt.subplot(2, 2, 2)
        plt.plot(history['data_loss'], label='Train')
        if 'val_data_loss' in history and history['val_data_loss']:
            plt.plot(history['val_data_loss'], label='Validation')
        plt.title('Data Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot physics loss
        plt.subplot(2, 2, 3)
        plt.plot(history['physics_loss'], label='Train')
        if 'val_physics_loss' in history and history['val_physics_loss']:
            plt.plot(history['val_physics_loss'], label='Validation')
        plt.title('Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_checkpoint(self, path, epoch, loss):
        """Save a model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint.
            epoch (int): Current epoch number.
            loss (float): Current loss value.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_model(self, checkpoint_path):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            
        Returns:
            dict: Checkpoint data.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Model loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
        
        return checkpoint 