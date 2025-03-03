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
from utils.visualization import (
    plot_pendulum_trajectory, 
    animate_pendulum, 
    create_comparison_directory,
    generate_random_predictions
)
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
    parser.add_argument('--physics_weight', type=float, default=0.7,
                        help='Weight for physics-informed loss (between 0 and 1)')
    
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate if creating data')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (epochs)')
    parser.add_argument('--num_test_trajectories', type=int, default=3,
                        help='Number of test trajectories to generate for comparison')
    
    return parser.parse_args()


def generate_ground_truth_trajectory(initial_theta, initial_omega, g=9.81, L=1.0, num_steps=200, dt=0.05):
    """Generate ground truth pendulum trajectory using physical equations.
    
    Args:
        initial_theta (float): Initial angle in radians.
        initial_omega (float): Initial angular velocity in radians/second.
        g (float): Gravitational acceleration (m/s^2).
        L (float): Pendulum length (m).
        num_steps (int): Number of time steps to generate.
        dt (float): Time step size in seconds.
        
    Returns:
        tuple: (times, thetas, omegas) arrays containing the trajectory.
    """
    # Initialize arrays
    times = np.linspace(0, dt * num_steps, num_steps)
    thetas = np.zeros(num_steps)
    omegas = np.zeros(num_steps)
    
    # Set initial conditions
    thetas[0] = initial_theta
    omegas[0] = initial_omega
    
    # Euler integration of the pendulum equations
    for i in range(1, num_steps):
        # Compute acceleration (second derivative of theta)
        alpha = -g / L * np.sin(thetas[i-1])
        
        # Update velocity (first derivative of theta)
        omegas[i] = omegas[i-1] + alpha * dt
        
        # Update position (theta)
        thetas[i] = thetas[i-1] + omegas[i] * dt
    
    return times, thetas, omegas


def main():
    """Main function."""
    args = parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create a comparison directory for visualizations
    comp_dir = create_comparison_directory("results/pendulum")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load or generate dataset
    if args.data_path:
        print(f"Loading dataset from {args.data_path}")
        dataset = PendulumDataset(data_path=args.data_path)
    else:
        print(f"Generating pendulum dataset with {args.samples} samples")
        dataset = PendulumDataset(generate=True, samples=args.samples)
        dataset.save_dataset(args.save_data_path)
    
    # Split dataset into train and validation sets
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    print(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}")
    
    # Initialize model
    model = PendulumPINN(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    # Create an untrained model (same architecture but different random initialization)
    untrained_model = PendulumPINN(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    # Move models to device
    model.to(device)
    untrained_model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize trainer
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
    
    # Train model
    print("Starting training...")
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir,
        save_freq=args.save_freq
    )
    print("Training completed.")
    
    # Generate comparisons for a few test trajectories
    print(f"Generating comparisons for {args.num_test_trajectories} test trajectories...")
    
    for i in range(args.num_test_trajectories):
        # Generate random initial conditions for test trajectory
        initial_theta = np.random.uniform(-np.pi/3, np.pi/3)
        initial_omega = np.random.uniform(-1.0, 1.0)
        
        # Generate ground truth trajectory
        times, true_thetas, true_omegas = generate_ground_truth_trajectory(
            initial_theta, initial_omega, num_steps=200, dt=0.05
        )
        
        # Get predictions from trained model
        pred_times, pred_thetas, pred_omegas = model.predict_trajectory(
            initial_theta, initial_omega, num_steps=len(times), dt=times[1]-times[0]
        )
        
        # Get predictions from untrained model
        untrained_times, untrained_thetas, untrained_omegas = untrained_model.predict_trajectory(
            initial_theta, initial_omega, num_steps=len(times), dt=times[1]-times[0]
        )
        
        # Generate random predictions
        random_thetas = generate_random_predictions(
            initial_theta, len(times), range_min=-np.pi, range_max=np.pi
        )
        random_omegas = generate_random_predictions(
            initial_omega, len(times), range_min=-3.0, range_max=3.0
        )
        
        # Compute physical errors
        true_errors = compute_pendulum_errors(times, true_thetas, true_omegas)
        pred_errors = compute_pendulum_errors(pred_times, pred_thetas, pred_omegas)
        untrained_errors = compute_pendulum_errors(untrained_times, untrained_thetas, untrained_omegas)
        random_errors = compute_pendulum_errors(times, random_thetas, random_omegas)
        
        # Calculate average error across all metrics for a single summary value
        true_avg_error = (true_errors['theta_error_mean'] + true_errors['omega_error_mean'] + true_errors['energy_error_mean']) / 3
        pred_avg_error = (pred_errors['theta_error_mean'] + pred_errors['omega_error_mean'] + pred_errors['energy_error_mean']) / 3
        untrained_avg_error = (untrained_errors['theta_error_mean'] + untrained_errors['omega_error_mean'] + untrained_errors['energy_error_mean']) / 3
        random_avg_error = (random_errors['theta_error_mean'] + random_errors['omega_error_mean'] + random_errors['energy_error_mean']) / 3
        
        print(f"\nTest trajectory {i+1}:")
        print(f"  Initial conditions: theta={initial_theta:.4f}, omega={initial_omega:.4f}")
        print(f"  Average physical error (ground truth): {true_avg_error:.6f}")
        print(f"  Average physical error (trained model): {pred_avg_error:.6f}")
        print(f"  Average physical error (untrained model): {untrained_avg_error:.6f}")
        print(f"  Average physical error (random prediction): {random_avg_error:.6f}")
        
        # Print detailed errors for the trained model
        print(f"  Detailed errors for trained model:")
        print(f"    Theta error: {pred_errors['theta_error_mean']:.6f} ± {pred_errors['theta_error_std']:.6f}")
        print(f"    Omega error: {pred_errors['omega_error_mean']:.6f} ± {pred_errors['omega_error_std']:.6f}")
        print(f"    Energy error: {pred_errors['energy_error_mean']:.6f} ± {pred_errors['energy_error_std']:.6f}")
        
        # Plot trajectory comparison
        plot_pendulum_trajectory(
            times, pred_thetas, pred_omegas,
            true_times=times, true_thetas=true_thetas, true_omegas=true_omegas,
            untrained_thetas=untrained_thetas, untrained_omegas=untrained_omegas,
            random_thetas=random_thetas, random_omegas=random_omegas,
            title=f"Pendulum Trajectory {i+1}",
            save_path=os.path.join(comp_dir, f"pendulum_trajectory_comparison_{i+1}.png")
        )
        
        # Create animation
        animate_pendulum(
            times, pred_thetas, L=1.0, 
            true_thetas=true_thetas,
            untrained_thetas=untrained_thetas,
            random_thetas=random_thetas,
            title=f"Pendulum Motion Comparison {i+1}",
            fps=30,
            save_path=os.path.join(comp_dir, f"pendulum_animation_comparison_{i+1}.gif")
        )
    
    print(f"\nComparisons saved to {comp_dir}")


if __name__ == "__main__":
    main() 