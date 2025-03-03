#!/usr/bin/env python3
"""Example script for training a physics-informed neural network for wave propagation."""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders.base import WaveDataset
from models.pinn import WavePINN
from training.trainer import PhysicsTrainer
from utils.visualization import plot_wave_field, animate_wave_field
from utils.evaluation import compute_wave_errors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a wave propagation PINN model')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to existing dataset (if not specified, data will be generated)')
    parser.add_argument('--save_data_path', type=str, default='data/datasets/wave_data.npz',
                        help='Path to save generated dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/wave',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/wave',
                        help='Directory for TensorBoard logs')
    
    parser.add_argument('--grid_size', type=int, default=32,
                        help='Size of the spatial grid (grid_size x grid_size)')
    parser.add_argument('--wave_speed', type=float, default=0.2,
                        help='Wave propagation speed')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of time steps to simulate per scenario')
    
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers in the model')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of hidden layers in the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--physics_weight', type=float, default=0.5,
                        help='Weight for physics-informed loss (between 0 and 1)')
    
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (epochs)')
    
    return parser.parse_args()


def main():
    """Train a wave propagation physics-informed neural network."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
    
    # Load or generate dataset
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        dataset = WaveDataset(data_path=args.data_path)
    else:
        print(f"Generating wave dataset with grid size {args.grid_size} and {args.samples} time steps")
        dataset = WaveDataset(
            generate=True,
            grid_size=args.grid_size,
            samples=args.samples,
            save_path=args.save_data_path
        )
    
    # Split dataset into training and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda
    )
    
    # Create model
    model = WavePINN(
        grid_size=args.grid_size,
        wave_speed=args.wave_speed,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        physics_weight=args.physics_weight,
        device=device,
        log_dir=args.log_dir
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir,
        save_freq=args.save_freq
    )
    
    # Plot loss history
    trainer.plot_history(
        history,
        save_path=os.path.join(args.checkpoint_dir, 'loss_history.png')
    )
    
    # Generate wave propagation prediction
    print("\nGenerating wave propagation prediction...")
    
    # Get a sample from the validation set
    inputs, _ = next(iter(val_loader))
    sample_input = inputs[0].to(device)  # Take the first sample
    
    # Predict the next 10 time steps
    model.eval()
    wave_fields = [sample_input.cpu().numpy()]
    
    current_wave = sample_input
    with torch.no_grad():
        for _ in range(10):
            next_wave = model(current_wave.unsqueeze(0)).squeeze(0)
            wave_fields.append(next_wave.cpu().numpy())
            current_wave = next_wave
    
    wave_fields = np.array(wave_fields)
    
    # Compute wave equation errors
    errors = compute_wave_errors(
        wave_fields,
        dt=model.dt,
        dx=model.dx,
        c=args.wave_speed
    )
    
    print("\nPhysical error metrics:")
    for key, value in errors.items():
        print(f"{key}: {value:.6f}")
    
    # Plot wave fields
    for i, wave_field in enumerate(wave_fields):
        plot_wave_field(
            wave_field,
            grid_size=args.grid_size,
            title=f"Wave Field (Time Step {i})",
            save_path=os.path.join(args.checkpoint_dir, f'wave_field_{i}.png')
        )
    
    # Create animation
    animate_wave_field(
        wave_fields,
        grid_size=args.grid_size,
        title="Wave Propagation",
        fps=4,
        save_path=os.path.join(args.checkpoint_dir, 'wave_animation.gif')
    )
    
    print(f"\nTraining complete! Model saved to {args.checkpoint_dir}")
    print(f"Check the wave field plots and animation in {args.checkpoint_dir}")


if __name__ == "__main__":
    main() 