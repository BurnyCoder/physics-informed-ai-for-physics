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
from utils.visualization import plot_pendulum_trajectory, animate_pendulum, create_comparison_directory
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
    parser.add_argument('--num_test_trajectories', type=int, default=3,
                        help='Number of test trajectories to generate for comparison')
    
    return parser.parse_args()


def generate_ground_truth_trajectory(initial_theta, initial_omega, g=9.81, L=1.0, num_steps=200, dt=0.05):
    """Generate ground truth pendulum trajectory using exact physics.
    
    Args:
        initial_theta (float): Initial angle in radians.
        initial_omega (float): Initial angular velocity in radians/s.
        g (float): Gravitational acceleration in m/s^2.
        L (float): Pendulum length in meters.
        num_steps (int): Number of time steps to generate.
        dt (float): Time step size in seconds.
        
    Returns:
        tuple: (times, thetas, omegas) with the ground truth trajectory.
    """
    # Initialize arrays
    times = np.zeros(num_steps)
    thetas = np.zeros(num_steps)
    omegas = np.zeros(num_steps)
    
    # Set initial conditions
    thetas[0] = initial_theta
    omegas[0] = initial_omega
    
    # Solve using Euler integration (simple but sufficient for demonstration)
    for i in range(1, num_steps):
        times[i] = i * dt
        
        # Update angular velocity using physics equation: dω/dt = -(g/L)sin(θ)
        omegas[i] = omegas[i-1] - (g/L) * np.sin(thetas[i-1]) * dt
        
        # Update angle using updated angular velocity: dθ/dt = ω
        thetas[i] = thetas[i-1] + omegas[i] * dt
    
    return times, thetas, omegas


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
    
    # Create comparison directory
    comp_dir = create_comparison_directory(args.checkpoint_dir)
    
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
    
    # Generate test trajectories for comparisons
    print(f"\nGenerating {args.num_test_trajectories} test trajectories for comparison...")
    
    # Test with different initial conditions
    test_conditions = [
        (0.5, 0.0),    # Moderate angle, no initial velocity
        (1.2, 0.0),    # Large angle, no initial velocity
        (0.3, 0.8),    # Small angle with initial velocity
        (-0.7, -0.5),  # Negative angle with negative velocity
        (1.5, -1.0)    # Large angle with negative velocity
    ]
    
    # Use only the number of test trajectories specified
    test_conditions = test_conditions[:min(args.num_test_trajectories, len(test_conditions))]
    
    for i, (initial_theta, initial_omega) in enumerate(test_conditions):
        print(f"\nTest trajectory {i+1}:")
        print(f"Initial angle: {initial_theta:.2f} radians")
        print(f"Initial angular velocity: {initial_omega:.2f} radians/s")
        
        # Generate ground truth trajectory
        true_times, true_thetas, true_omegas = generate_ground_truth_trajectory(
            initial_theta=initial_theta,
            initial_omega=initial_omega,
            num_steps=200,
            dt=0.05
        )
        
        # Generate model prediction
        pred_times, pred_thetas, pred_omegas = model.predict_trajectory(
            initial_theta=initial_theta,
            initial_omega=initial_omega,
            num_steps=200,
            dt=0.05
        )
        
        # Compute physical errors for prediction
        errors = compute_pendulum_errors(pred_times, pred_thetas, pred_omegas)
        
        print("\nPhysical error metrics:")
        for key, value in errors.items():
            print(f"{key}: {value:.6f}")
        
        # Plot comparison trajectory
        plot_pendulum_trajectory(
            times=pred_times, 
            thetas=pred_thetas, 
            omegas=pred_omegas,
            true_times=true_times,
            true_thetas=true_thetas,
            true_omegas=true_omegas,
            title=f"Pendulum Trajectory {i+1}",
            save_path=os.path.join(comp_dir, f'pendulum_trajectory_comparison_{i+1}.png')
        )
        
        # Generate comparison animation
        animate_pendulum(
            times=pred_times,
            thetas=pred_thetas,
            true_thetas=true_thetas,
            title=f"Pendulum Motion Comparison {i+1}",
            fps=30,
            save_path=os.path.join(comp_dir, f'pendulum_animation_comparison_{i+1}.gif')
        )
    
    print(f"\nTraining complete! Model saved to {args.checkpoint_dir}")
    print(f"Check the trajectory plots and animations in {comp_dir}")


if __name__ == "__main__":
    main() 