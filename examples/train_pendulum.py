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
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for output
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
    comparison_dir = create_comparison_directory('comparisons/pendulum')
    
    # Load or generate dataset
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        dataset = PendulumDataset(data_path=args.data_path)
    else:
        print(f"Generating new pendulum dataset with {args.samples} samples")
        dataset = PendulumDataset(generate=True, samples=args.samples)
        dataset.save_dataset(args.save_data_path)
        print(f"Saved dataset to {args.save_data_path}")
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create the model and untrained model (for comparison)
    model = PendulumPINN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    untrained_model = PendulumPINN(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
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
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir,
        save_freq=args.save_freq
    )
    
    # Generate predictions for a test sample from validation set
    num_test = args.num_test_trajectories if hasattr(args, 'num_test_trajectories') else 3
    
    for test_idx in range(min(num_test, len(val_dataset))):
        print(f"\nGenerating predictions for test sample {test_idx+1}/{num_test}")
        
        # Get test sample
        inputs, targets = val_dataset[test_idx]
        initial_theta = inputs[0].item()
        initial_omega = inputs[1].item()
        
        # Generate ground truth trajectory
        print(f"Ground truth trajectory from θ₀={initial_theta:.4f}, ω₀={initial_omega:.4f}")
        true_times, true_thetas, true_omegas = generate_ground_truth_trajectory(
            initial_theta, initial_omega, num_steps=200, dt=0.05
        )
        
        # Generate trajectory from trained model
        pred_times, pred_thetas, pred_omegas = model.predict_trajectory(
            initial_theta, initial_omega, num_steps=200, dt=0.05
        )
        
        # Generate trajectory from untrained model
        untrained_times, untrained_thetas, untrained_omegas = untrained_model.predict_trajectory(
            initial_theta, initial_omega, num_steps=200, dt=0.05
        )
        
        # Generate random trajectory (for baseline comparison)
        random_thetas = generate_random_predictions(initial_theta, len(true_times))
        random_omegas = generate_random_predictions(initial_omega, len(true_times))
        
        # Compute errors
        true_errors = compute_pendulum_errors(true_times, true_thetas, true_omegas)
        pred_errors = compute_pendulum_errors(pred_times, pred_thetas, pred_omegas)
        untrained_errors = compute_pendulum_errors(untrained_times, untrained_thetas, untrained_omegas)
        random_errors = compute_pendulum_errors(true_times, random_thetas, random_omegas)
        
        # Calculate average errors
        def calculate_avg_error(error_dict):
            """Calculate average of all error metrics ending with '_mean'."""
            mean_errors = [v for k, v in error_dict.items() if k.endswith('_mean')]
            return sum(mean_errors) / len(mean_errors) if mean_errors else 0
        
        true_avg_error = calculate_avg_error(true_errors)
        pred_avg_error = calculate_avg_error(pred_errors)
        untrained_avg_error = calculate_avg_error(untrained_errors)
        random_avg_error = calculate_avg_error(random_errors)
        
        # Print detailed errors for the trained model
        print("\nDetailed errors for trained model:")
        for key in [k for k in pred_errors.keys() if k.endswith('_mean')]:
            std_key = key.replace('_mean', '_std')
            metric_name = key.replace('_error_mean', '').capitalize()
            print(f"  {metric_name}: {pred_errors[key]:.6f} ± {pred_errors[std_key]:.6f}")
        
        # Print error comparison
        print("\nAverage error comparison:")
        print(f"  Ground truth: {true_avg_error:.6f}")
        print(f"  Trained model: {pred_avg_error:.6f}")
        print(f"  Untrained model: {untrained_avg_error:.6f}")
        print(f"  Random predictions: {random_avg_error:.6f}")
        
        improvement_over_untrained = (untrained_avg_error - pred_avg_error) / untrained_avg_error * 100
        improvement_over_random = (random_avg_error - pred_avg_error) / random_avg_error * 100
        
        print(f"\nImprovement over untrained model: {improvement_over_untrained:.2f}%")
        print(f"Improvement over random predictions: {improvement_over_random:.2f}%")
        
        # Save plots
        plot_save_path = os.path.join(comparison_dir, f'pendulum_trajectory_comparison_{test_idx+1}.png')
        print(f"\nGenerating comparison plot: {plot_save_path}")
        
        plot_pendulum_trajectory(
            pred_times, pred_thetas, pred_omegas,
            true_times=true_times, true_thetas=true_thetas, true_omegas=true_omegas,
            untrained_thetas=untrained_thetas, untrained_omegas=untrained_omegas,
            random_thetas=random_thetas, random_omegas=random_omegas,
            title=f"Pendulum Trajectory Comparison (Test Sample {test_idx+1})",
            save_path=plot_save_path
        )
        
        # Save animation
        animation_save_path = os.path.join(comparison_dir, f'pendulum_animation_comparison_{test_idx+1}.gif')
        print(f"Generating comparison animation: {animation_save_path}")
        
        animate_pendulum(
            pred_times, pred_thetas,
            true_thetas=true_thetas,
            untrained_thetas=untrained_thetas,
            random_thetas=random_thetas,
            title=f"Pendulum Motion Comparison (Test Sample {test_idx+1})",
            fps=30, save_path=animation_save_path
        )
        
        print(f"Saved visualizations for test sample {test_idx+1} to {comparison_dir}")
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main() 