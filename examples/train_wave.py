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
from utils.visualization import plot_wave_field, animate_wave_field, create_comparison_directory
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
    
    # Create comparison directory
    comp_dir = create_comparison_directory(args.checkpoint_dir)
    
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
    
    # Generate wave propagation comparison
    print("\nGenerating wave propagation comparison...")
    
    # Get 3 different samples from the validation set for testing
    num_test_samples = min(3, len(val_dataset))
    test_indices = np.random.choice(len(val_dataset), num_test_samples, replace=False)
    
    for i, idx in enumerate(test_indices):
        # Get a test sample
        inputs, _ = val_dataset[idx]
        sample_input = inputs.to(device)
        
        # Generate ground truth propagation
        print(f"\nGenerating ground truth and predictions for test sample {i+1}...")
        
        # Convert input tensor to numpy array for ground truth simulation
        initial_wave = sample_input.cpu().numpy()
        
        # Generate ground truth wave propagation
        true_wave_fields = propagate_wave_ground_truth(
            initial_wave=initial_wave,
            grid_size=args.grid_size,
            num_steps=args.num_prediction_steps,
            wave_speed=args.wave_speed
        )
        
        # Predict the next steps using the model
        model.eval()
        pred_wave_fields = [initial_wave]
        
        current_wave = sample_input
        with torch.no_grad():
            for _ in range(1, args.num_prediction_steps):
                next_wave = model(current_wave.unsqueeze(0)).squeeze(0)
                pred_wave_fields.append(next_wave.cpu().numpy())
                current_wave = next_wave
        
        pred_wave_fields = np.array(pred_wave_fields)
        
        # Compute wave equation errors
        errors = compute_wave_errors(
            pred_wave_fields,
            dt=model.dt,
            dx=model.dx,
            c=args.wave_speed
        )
        
        print(f"\nPhysical error metrics for test sample {i+1}:")
        for key, value in errors.items():
            print(f"{key}: {value:.6f}")
        
        # Plot wave fields at specific time steps
        plot_steps = [0, min(5, args.num_prediction_steps-1), args.num_prediction_steps-1]
        
        for step in plot_steps:
            plot_wave_field(
                wave_field=pred_wave_fields[step],
                grid_size=args.grid_size,
                title=f"Wave Field (Step {step})",
                true_wave_field=true_wave_fields[step],
                save_path=os.path.join(comp_dir, f'wave_field_comparison_{i+1}_step_{step}.png')
            )
        
        # Create animation
        animate_wave_field(
            wave_fields=pred_wave_fields,
            grid_size=args.grid_size,
            title=f"Wave Propagation Comparison {i+1}",
            fps=4,
            true_wave_fields=true_wave_fields,
            save_path=os.path.join(comp_dir, f'wave_animation_comparison_{i+1}.gif')
        )
    
    print(f"\nTraining complete! Model saved to {args.checkpoint_dir}")
    print(f"Check the wave field plots and animations in {comp_dir}")


if __name__ == "__main__":
    main() 