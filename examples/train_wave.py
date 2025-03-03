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

from data.loaders.wave import WaveDataset
from models.pinn import WavePINN
from training.trainer import PhysicsTrainer
from utils.visualization import (
    plot_wave_field, 
    animate_wave_field, 
    create_comparison_directory,
    generate_random_predictions
)
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
    parser.add_argument('--physics_weight', type=float, default=0.7,
                        help='Weight for physics-informed loss (between 0 and 1)')
    
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints (epochs)')
    parser.add_argument('--num_prediction_steps', type=int, default=20,
                        help='Number of steps to predict for comparison')
    
    return parser.parse_args()


def propagate_wave_ground_truth(initial_wave, grid_size, num_steps, wave_speed=0.2, dt=0.01, dx=0.02):
    """Propagate wave using ground truth wave equation.
    
    Args:
        initial_wave (ndarray): Initial wave field (2D grid).
        grid_size (int): Size of the grid (grid_size x grid_size).
        num_steps (int): Number of time steps to simulate.
        wave_speed (float): Speed of wave propagation.
        dt (float): Time step size.
        dx (float): Spatial grid size.
        
    Returns:
        ndarray: Wave fields over time of shape (num_steps, grid_size, grid_size).
    """
    # Make sure the initial wave is a 2D grid
    if initial_wave.ndim == 1:
        initial_wave = initial_wave.reshape(grid_size, grid_size)
    
    # Need previous and current state to compute next state
    # For initial step, assume the wave was stationary (previous = current)
    prev_wave = initial_wave.copy()
    current_wave = initial_wave.copy()
    
    # Initialize array to store all wave fields
    wave_fields = np.zeros((num_steps, grid_size, grid_size))
    wave_fields[0] = initial_wave
    
    # Square of wave speed times square of time step divided by square of space step
    c_squared = wave_speed ** 2
    factor = c_squared * (dt/dx) ** 2
    
    # Propagate the wave
    for t in range(1, num_steps):
        # Create a copy of the current wave for the next step
        next_wave = np.zeros_like(current_wave)
        
        # Apply wave equation to interior points
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                # Wave equation in discrete form
                # u_{i,j}^{n+1} = 2*u_{i,j}^n - u_{i,j}^{n-1} + c^2 * (dt/dx)^2 * 
                # (u_{i+1,j}^n + u_{i-1,j}^n + u_{i,j+1}^n + u_{i,j-1}^n - 4*u_{i,j}^n)
                laplacian = (
                    current_wave[i+1, j] + current_wave[i-1, j] +
                    current_wave[i, j+1] + current_wave[i, j-1] - 
                    4 * current_wave[i, j]
                )
                
                next_wave[i, j] = (
                    2 * current_wave[i, j] - prev_wave[i, j] + 
                    factor * laplacian
                )
        
        # Apply boundary conditions (fixed boundaries)
        next_wave[0, :] = next_wave[-1, :] = next_wave[:, 0] = next_wave[:, -1] = 0
        
        # Store the wave field
        wave_fields[t] = next_wave
        
        # Update for next iteration
        prev_wave = current_wave.copy()
        current_wave = next_wave.copy()
    
    return wave_fields


def generate_random_wave_fields(initial_wave, grid_size, num_steps, amplitude_range=(-1.0, 1.0), smooth_factor=0.8):
    """Generate random wave fields with some spatial and temporal smoothness.
    
    Args:
        initial_wave (array): Initial wave field.
        grid_size (int): Size of the grid.
        num_steps (int): Number of time steps to generate.
        amplitude_range (tuple): Range for wave amplitude values.
        smooth_factor (float): Smoothness factor between 0 and 1.
            - 0 means completely random
            - 1 means constant initial field
    
    Returns:
        array: Random wave fields of shape (num_steps, grid_size, grid_size).
    """
    # Reshape if needed
    if initial_wave.ndim == 1:
        initial_wave = initial_wave.reshape(grid_size, grid_size)
    
    # Initialize the wave fields array
    wave_fields = np.zeros((num_steps, grid_size, grid_size))
    wave_fields[0] = initial_wave.copy()
    
    # Smoothing kernel for spatial smoothing
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # Generate random fields with smoothness
    for t in range(1, num_steps):
        # Start with previous field
        wave_fields[t] = wave_fields[t-1].copy()
        
        # Add random noise
        random_field = np.random.uniform(
            amplitude_range[0], amplitude_range[1], 
            size=(grid_size, grid_size)
        )
        
        # Apply smoothing in space
        # Simple convolution for smoothness (edges handled by maintaining previous values)
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                # Extract local patch
                patch = random_field[i-1:i+2, j-1:j+2]
                # Apply kernel
                smoothed_value = np.sum(patch * kernel)
                # Update with weighted combination of previous and smoothed random
                wave_fields[t, i, j] = (
                    smooth_factor * wave_fields[t-1, i, j] + 
                    (1 - smooth_factor) * smoothed_value
                )
    
    return wave_fields


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for output
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
    comparison_dir = create_comparison_directory('comparisons/wave')
    
    # Load or generate dataset
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading dataset from {args.data_path}")
        dataset = WaveDataset(data_path=args.data_path)
    else:
        print(f"Generating new wave dataset with {args.samples} samples")
        dataset = WaveDataset(
            generate=True, 
            samples=args.samples,
            grid_size=args.grid_size,
            time_steps=args.time_steps,
            wave_speed=args.wave_speed
        )
        dataset.save_dataset(args.save_data_path)
        print(f"Saved dataset to {args.save_data_path}")
    
    # Split dataset into training and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create the model and untrained model (for comparison)
    model = WavePINN(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    # Create untrained model with same architecture
    untrained_model = WavePINN(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
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
    
    # Number of steps to predict
    num_prediction_steps = args.num_prediction_steps if hasattr(args, 'num_prediction_steps') else 20
    
    # Generate predictions for a test sample from validation set
    num_test_samples = min(3, len(val_dataset))
    
    for test_idx in range(num_test_samples):
        print(f"\nGenerating predictions for test sample {test_idx+1}/{num_test_samples}")
        
        # Get test sample
        inputs, targets = val_dataset[test_idx]
        initial_wave = inputs.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get parameters from dataset
        grid_size = dataset.grid_size
        dt = dataset.dt
        dx = dataset.dx
        wave_speed = dataset.wave_speed
        
        # Generate ground truth wave propagation
        true_wave_fields = propagate_wave_ground_truth(
            initial_wave.squeeze(0).cpu().numpy(),
            grid_size=grid_size,
            num_steps=num_prediction_steps,
            wave_speed=wave_speed,
            dt=dt,
            dx=dx
        )
        
        # Generate trained model predictions
        pred_wave_fields = model.predict_wave_propagation(
            initial_wave,
            num_steps=num_prediction_steps
        )
        pred_wave_fields = pred_wave_fields.cpu().numpy()
        
        # Generate untrained model predictions
        untrained_wave_fields = untrained_model.predict_wave_propagation(
            initial_wave, 
            num_steps=num_prediction_steps
        )
        untrained_wave_fields = untrained_wave_fields.cpu().numpy()
        
        # Generate random predictions
        random_wave_fields = np.zeros_like(true_wave_fields)
        for t in range(num_prediction_steps):
            # Generate random wave field based on initial range
            initial_min = initial_wave.cpu().numpy().min()
            initial_max = initial_wave.cpu().numpy().max()
            random_wave_fields[t] = np.random.uniform(
                initial_min, initial_max, size=(grid_size, grid_size)
            )
        
        # Compute errors
        true_errors = compute_wave_errors(true_wave_fields, true_wave_fields)
        pred_errors = compute_wave_errors(pred_wave_fields, true_wave_fields)
        untrained_errors = compute_wave_errors(untrained_wave_fields, true_wave_fields)
        random_errors = compute_wave_errors(random_wave_fields, true_wave_fields)
        
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
        
        # Save plots for specific timesteps
        for t in [0, num_prediction_steps // 2, num_prediction_steps - 1]:
            plot_save_path = os.path.join(
                comparison_dir, 
                f'wave_field_comparison_sample{test_idx+1}_t{t}.png'
            )
            
            print(f"\nGenerating comparison plot at timestep {t}: {plot_save_path}")
            
            plot_wave_field(
                pred_wave_fields[t],
                true_wave_field=true_wave_fields[t],
                untrained_wave_field=untrained_wave_fields[t],
                random_wave_field=random_wave_fields[t],
                title=f"Wave Field Comparison (Sample {test_idx+1}, t={t})",
                save_path=plot_save_path
            )
        
        # Save animation
        animation_save_path = os.path.join(
            comparison_dir, 
            f'wave_animation_comparison_sample{test_idx+1}.gif'
        )
        
        print(f"Generating comparison animation: {animation_save_path}")
        
        animate_wave_field(
            pred_wave_fields,
            true_wave_fields=true_wave_fields,
            untrained_wave_fields=untrained_wave_fields,
            random_wave_fields=random_wave_fields,
            title=f"Wave Propagation Comparison (Sample {test_idx+1})",
            fps=10,
            save_path=animation_save_path
        )
        
        print(f"Saved visualizations for test sample {test_idx+1} to {comparison_dir}")
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main() 