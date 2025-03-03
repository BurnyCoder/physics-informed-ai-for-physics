#!/usr/bin/env python3
"""Example script for training a physics-informed neural network for pendulum motion."""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders.base import PendulumDataset
from models.pinn import PendulumPINN
from training.trainer import PhysicsTrainer
from utils.visualization import plot_pendulum_trajectory, animate_pendulum
from utils.evaluation import compute_pendulum_errors


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a pendulum PINN model')
    
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to existing dataset (if not specified, data will be generated)')
    parser.add_argument('--save_data_path', type=str, default='data/datasets/pendulum_data.npz',
                        help='Path to save generated dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension of hidden layers in the model')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of hidden layers in the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--physics_weight', type=float, default=0.5,
                        help='Weight for physics-informed loss (between 0 and 1)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate if creating data')
    
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (epochs)')
    
    return parser.parse_args()


def main():
    """Train a pendulum physics-informed neural network."""
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
        dataset = PendulumDataset(data_path=args.data_path)
    else:
        print(f"Generating pendulum dataset with {args.samples} samples")
        dataset = PendulumDataset(generate=True, samples=args.samples)
        
        # Save the generated dataset
        dataset.save_dataset(args.save_data_path)
    
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
    model = PendulumPINN(
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
    
    # Evaluate on a test trajectory
    initial_theta = 0.5  # 0.5 radians initial angle
    initial_omega = 0.0  # 0 initial angular velocity
    
    print("\nGenerating test trajectory...")
    times, thetas, omegas = model.predict_trajectory(
        initial_theta=initial_theta,
        initial_omega=initial_omega,
        num_steps=200,
        dt=0.05
    )
    
    # Compute physical errors
    errors = compute_pendulum_errors(times, thetas, omegas)
    
    print("\nPhysical error metrics:")
    for key, value in errors.items():
        print(f"{key}: {value:.6f}")
    
    # Plot predicted trajectory
    plot_pendulum_trajectory(
        times, thetas, omegas,
        title="Pendulum Trajectory Prediction",
        save_path=os.path.join(args.checkpoint_dir, 'pendulum_trajectory.png')
    )
    
    # Generate animation
    animate_pendulum(
        times, thetas,
        title="Pendulum Motion",
        fps=30,
        save_path=os.path.join(args.checkpoint_dir, 'pendulum_animation.gif')
    )
    
    print(f"\nTraining complete! Model saved to {args.checkpoint_dir}")
    print(f"Check the trajectory plot and animation in {args.checkpoint_dir}")


if __name__ == "__main__":
    main() 