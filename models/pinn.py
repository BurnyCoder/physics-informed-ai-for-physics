"""Physics-Informed Neural Networks (PINNs) implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PINN(nn.Module):
    """Base class for Physics-Informed Neural Networks."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=4):
        """Initialize the PINN model.
        
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of hidden layers.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        return self.net(x)
    
    def physics_loss(self, x, y_pred):
        """Compute the physics-informed loss.
        
        This method should be implemented by subclasses to enforce physical laws.
        
        Args:
            x (torch.Tensor): Input tensor.
            y_pred (torch.Tensor): Predicted output tensor.
            
        Returns:
            torch.Tensor: Physics loss term.
        """
        raise NotImplementedError("Subclasses must implement physics_loss method")
    
    def compute_gradients(self, y, x, create_graph=True):
        """Compute gradients of y with respect to x using automatic differentiation.
        
        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor that requires gradients.
            create_graph (bool): Whether to create a computational graph of the derivatives.
            
        Returns:
            torch.Tensor: Gradients of y with respect to x.
        """
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        return gradients


class PendulumPINN(PINN):
    """Physics-Informed Neural Network for pendulum motion."""
    
    def __init__(self, g=9.81, L=1.0, hidden_dim=64, num_layers=4):
        """Initialize the pendulum PINN.
        
        Args:
            g (float): Gravitational acceleration (m/s^2).
            L (float): Pendulum length (m).
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of hidden layers.
        """
        # Input: [t, theta, omega]
        # Output: [theta_next, omega_next]
        super().__init__(input_dim=3, output_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
        
        self.g = g  # gravitational acceleration
        self.L = L  # pendulum length
    
    def physics_loss(self, x, y_pred):
        """Compute the physics-informed loss for pendulum motion.
        
        The pendulum follows the equation: d^2θ/dt^2 + (g/L)*sin(θ) = 0
        
        Args:
            x (torch.Tensor): Input tensor [t, theta, omega].
            y_pred (torch.Tensor): Predicted output tensor [theta_next, omega_next].
            
        Returns:
            torch.Tensor: Physics loss term.
        """
        # Extract current state variables
        t = x[:, 0].reshape(-1, 1)
        theta = x[:, 1].reshape(-1, 1)
        omega = x[:, 2].reshape(-1, 1)
        
        # Extract predicted next state
        theta_next = y_pred[:, 0].reshape(-1, 1)
        omega_next = y_pred[:, 1].reshape(-1, 1)
        
        # Time step (assuming uniform time steps)
        dt = 0.1  # This should match the dataset's time step
        
        # Euler integration for theta: theta_next ≈ theta + omega * dt
        theta_next_physics = theta + omega * dt
        
        # Euler integration for omega: omega_next ≈ omega - (g/L)*sin(theta) * dt
        omega_next_physics = omega - (self.g / self.L) * torch.sin(theta) * dt
        
        # Physics loss is the mean squared error between predicted and physics-based values
        theta_loss = F.mse_loss(theta_next, theta_next_physics)
        omega_loss = F.mse_loss(omega_next, omega_next_physics)
        
        return theta_loss + omega_loss
    
    def predict_trajectory(self, initial_theta, initial_omega, num_steps=100, dt=0.1):
        """Predict the pendulum trajectory given initial conditions.
        
        Args:
            initial_theta (float): Initial angle (radians).
            initial_omega (float): Initial angular velocity (radians/s).
            num_steps (int): Number of time steps to predict.
            dt (float): Time step size.
            
        Returns:
            tuple: (times, thetas, omegas) numpy arrays with the predicted trajectory.
        """
        self.eval()  # Set to evaluation mode
        
        # Initialize arrays to store the trajectory
        times = np.zeros(num_steps)
        thetas = np.zeros(num_steps)
        omegas = np.zeros(num_steps)
        
        # Set initial conditions
        thetas[0] = initial_theta
        omegas[0] = initial_omega
        
        # Current state
        t = 0.0
        theta = initial_theta
        omega = initial_omega
        
        with torch.no_grad():
            for i in range(1, num_steps):
                # Update time
                t += dt
                times[i] = t
                
                # Prepare input for the model
                x = torch.tensor([[t, theta, omega]], dtype=torch.float32)
                
                # Get model prediction
                y_pred = self(x)
                
                # Update state
                theta = y_pred[0, 0].item()
                omega = y_pred[0, 1].item()
                
                # Store predictions
                thetas[i] = theta
                omegas[i] = omega
        
        return times, thetas, omegas


class WavePINN(PINN):
    """Physics-Informed Neural Network for wave propagation."""
    
    def __init__(self, grid_size=50, wave_speed=0.2, hidden_dim=128, num_layers=6):
        """Initialize the wave PINN.
        
        Args:
            grid_size (int): Size of the spatial grid (grid_size x grid_size).
            wave_speed (float): Speed of wave propagation.
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of hidden layers.
        """
        # Input: flattened wave field (grid_size^2)
        # Output: flattened wave field at next time step (grid_size^2)
        super().__init__(
            input_dim=grid_size**2, 
            output_dim=grid_size**2, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers
        )
        
        self.grid_size = grid_size
        self.wave_speed = wave_speed
        self.c_squared = wave_speed**2
        
        # Spatial step
        self.dx = 1.0 / (grid_size - 1)
        
        # Time step (for stability)
        self.dt = 0.1 * self.dx / wave_speed
    
    def forward(self, x):
        """Forward pass with a residual connection to help with training stability.
        
        Args:
            x (torch.Tensor): Input tensor representing the wave field.
            
        Returns:
            torch.Tensor: Predicted next wave field.
        """
        # Using a residual connection helps with learning small changes
        # between consecutive wave fields
        delta = self.net(x)
        return x + delta
    
    def physics_loss(self, x, y_pred):
        """Compute the physics-informed loss for wave propagation.
        
        For the wave equation: ∂²u/∂t² = c² * (∂²u/∂x² + ∂²u/∂y²)
        
        Args:
            x (torch.Tensor): Input tensor representing current wave field.
            y_pred (torch.Tensor): Predicted output tensor representing next wave field.
            
        Returns:
            torch.Tensor: Physics loss term.
        """
        batch_size = x.shape[0]
        
        # Reshape tensors to 2D grids
        u = x.reshape(batch_size, self.grid_size, self.grid_size)
        u_next = y_pred.reshape(batch_size, self.grid_size, self.grid_size)
        
        # Compute spatial derivatives using finite differences
        # For each sample in the batch
        physics_loss = 0.0
        
        for i in range(batch_size):
            # Extract current sample
            u_i = u[i]
            u_next_i = u_next[i]
            
            # Skip boundary points (assumed to be zero)
            for y in range(1, self.grid_size-1):
                for x in range(1, self.grid_size-1):
                    # Compute Laplacian at point (x,y)
                    laplacian = (
                        u_i[y+1, x] + u_i[y-1, x] + 
                        u_i[y, x+1] + u_i[y, x-1] - 
                        4 * u_i[y, x]
                    ) / (self.dx**2)
                    
                    # Get previous time step (we don't have it directly, so we approximate)
                    # This is a limitation, ideally we'd have both u(t) and u(t-1) in our input
                    u_prev_yx = u_i[y, x] - (u_next_i[y, x] - u_i[y, x])
                    
                    # Wave equation discrete form: u_next = 2*u - u_prev + c²*dt²*laplacian
                    u_next_physics = 2 * u_i[y, x] - u_prev_yx + self.c_squared * self.dt**2 * laplacian
                    
                    # Accumulate squared errors
                    physics_loss += (u_next_i[y, x] - u_next_physics)**2
        
        # Normalize by number of interior points
        num_interior_points = (self.grid_size - 2)**2 * batch_size
        return physics_loss / num_interior_points 