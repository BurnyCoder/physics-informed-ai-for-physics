"""Evaluation utilities for physics-based AI."""

import numpy as np
import torch


def compute_physics_error(model, inputs, predictions, physics_type='pendulum', **kwargs):
    """Compute how well the predictions adhere to physical laws.
    
    Args:
        model: Physics-informed neural network model.
        inputs: Input tensor or numpy array.
        predictions: Predicted output tensor or numpy array.
        physics_type (str): Type of physics problem ('pendulum', 'wave', etc.).
        **kwargs: Additional parameters for specific physics types.
        
    Returns:
        float: Mean physics error.
    """
    # Convert numpy arrays to tensors if needed
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    
    # Make sure gradients are not computed
    with torch.no_grad():
        physics_loss = model.physics_loss(inputs, predictions)
    
    return physics_loss.item()


def compute_pendulum_errors(times, thetas, omegas, g=9.81, L=1.0):
    """Compute errors in pendulum motion with respect to physical laws.
    
    Args:
        times: Array of time points.
        thetas: Array of pendulum angles.
        omegas: Array of angular velocities.
        g: Gravitational acceleration (m/s^2).
        L: Pendulum length (m).
        
    Returns:
        dict: Dictionary with error metrics.
    """
    # Time step
    dt = times[1] - times[0] if len(times) > 1 else 0.1
    
    # Initialize errors
    theta_errors = []
    omega_errors = []
    energy_errors = []
    
    # Initial energy (should be conserved for undamped pendulum)
    E0 = 0.5 * (omegas[0]**2) + g/L * (1 - np.cos(thetas[0]))
    
    # Check physical consistency
    for i in range(1, len(times) - 1):
        # Compute theta error: dθ/dt = ω
        dtheta_dt = (thetas[i+1] - thetas[i-1]) / (2 * dt)
        theta_error = np.abs(dtheta_dt - omegas[i])
        theta_errors.append(theta_error)
        
        # Compute omega error: dω/dt = -(g/L)sin(θ)
        domega_dt = (omegas[i+1] - omegas[i-1]) / (2 * dt)
        omega_expected = -(g/L) * np.sin(thetas[i])
        omega_error = np.abs(domega_dt - omega_expected)
        omega_errors.append(omega_error)
        
        # Compute energy error (normalized)
        E = 0.5 * (omegas[i]**2) + g/L * (1 - np.cos(thetas[i]))
        energy_error = np.abs((E - E0) / E0) if E0 != 0 else np.abs(E)
        energy_errors.append(energy_error)
    
    return {
        'theta_error_mean': np.mean(theta_errors),
        'theta_error_std': np.std(theta_errors),
        'omega_error_mean': np.mean(omega_errors),
        'omega_error_std': np.std(omega_errors),
        'energy_error_mean': np.mean(energy_errors),
        'energy_error_std': np.std(energy_errors)
    }


def compute_wave_errors(wave_fields, dt=0.1, dx=0.02, c=0.2):
    """Compute errors in wave propagation with respect to physical laws.
    
    Args:
        wave_fields: Array of wave fields over time of shape (num_frames, height, width)
                    or (num_frames, flattened_grid).
        dt: Time step between frames.
        dx: Spatial step.
        c: Wave speed.
        
    Returns:
        dict: Dictionary with error metrics.
    """
    # Make sure wave_fields is 3D
    if wave_fields.ndim == 2:
        # Assume square grid
        grid_size = int(np.sqrt(wave_fields.shape[1]))
        wave_fields = wave_fields.reshape(-1, grid_size, grid_size)
    
    num_frames, height, width = wave_fields.shape
    
    # Initialize error metrics
    wave_equation_errors = []
    
    # Compute errors based on wave equation: ∂²u/∂t² = c² * (∂²u/∂x² + ∂²u/∂y²)
    for i in range(1, num_frames - 1):
        # Current wave field
        u = wave_fields[i]
        
        # Time derivatives (second order central difference)
        d2u_dt2 = (wave_fields[i+1] + wave_fields[i-1] - 2*u) / (dt**2)
        
        # Initialize spatial Laplacian
        laplacian = np.zeros_like(u)
        
        # Compute Laplacian using finite differences, skipping boundaries
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Compute spatial derivatives (second order central differences)
                d2u_dx2 = (u[y, x+1] + u[y, x-1] - 2*u[y, x]) / (dx**2)
                d2u_dy2 = (u[y+1, x] + u[y-1, x] - 2*u[y, x]) / (dx**2)
                
                laplacian[y, x] = d2u_dx2 + d2u_dy2
        
        # Wave equation: d2u_dt2 = c^2 * laplacian
        # Compute error as absolute difference
        error = np.abs(d2u_dt2 - c**2 * laplacian)
        
        # Average error over interior points (excluding boundaries)
        error_mean = np.mean(error[1:-1, 1:-1])
        wave_equation_errors.append(error_mean)
    
    return {
        'wave_equation_error_mean': np.mean(wave_equation_errors),
        'wave_equation_error_std': np.std(wave_equation_errors),
        'wave_equation_error_max': np.max(wave_equation_errors)
    }


def compute_model_accuracy(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Compute model accuracy on a test dataset.
    
    Args:
        model: Neural network model.
        test_loader: DataLoader with test data.
        device: Device to use for computation.
        
    Returns:
        dict: Dictionary with accuracy metrics.
    """
    model.eval()
    
    total_mse = 0.0
    total_physics_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            
            # Compute MSE
            mse = torch.nn.functional.mse_loss(outputs, targets).item()
            
            # Compute physics loss
            physics_loss = model.physics_loss(inputs, outputs).item()
            
            total_mse += mse
            total_physics_loss += physics_loss
    
    # Calculate average metrics
    avg_mse = total_mse / len(test_loader)
    avg_physics_loss = total_physics_loss / len(test_loader)
    
    return {
        'mse': avg_mse,
        'physics_loss': avg_physics_loss,
        'rmse': np.sqrt(avg_mse)
    } 