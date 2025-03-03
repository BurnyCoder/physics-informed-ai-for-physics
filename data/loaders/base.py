"""Base classes for loading physics datasets."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PhysicsDataset(Dataset):
    """Base class for physics datasets."""
    
    def __init__(self, data_path=None, transform=None):
        """Initialize the physics dataset.
        
        Args:
            data_path (str): Path to the dataset file.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        self.data_path = data_path
        self.transform = transform
        self.data = None
        self.targets = None
        
        if data_path and os.path.exists(data_path):
            self._load_data()
    
    def _load_data(self):
        """Load data from file. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_data method")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.data is None:
            return 0
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to return.
            
        Returns:
            tuple: (input_data, target) where target is the physics quantity to predict.
        """
        if self.data is None or self.targets is None:
            raise ValueError("Dataset is empty. Call _load_data or initialize properly.")
        
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, target


class PendulumDataset(PhysicsDataset):
    """Dataset for pendulum motion."""
    
    def __init__(self, data_path=None, transform=None, generate=False, samples=1000):
        """Initialize pendulum dataset.
        
        Args:
            data_path (str, optional): Path to the dataset file.
            transform (callable, optional): Transform to be applied to samples.
            generate (bool): Whether to generate synthetic data.
            samples (int): Number of samples to generate if generate=True.
        """
        self.generate = generate
        self.samples = samples
        super().__init__(data_path, transform)
        
        if generate:
            self._generate_data()
    
    def _load_data(self):
        """Load pendulum data from file."""
        try:
            data = np.load(self.data_path)
            self.data = torch.tensor(data['inputs'], dtype=torch.float32)
            self.targets = torch.tensor(data['targets'], dtype=torch.float32)
        except Exception as e:
            raise IOError(f"Failed to load pendulum data: {e}")
    
    def _generate_data(self):
        """Generate synthetic pendulum data."""
        # Parameters
        g = 9.81  # gravitational acceleration (m/s^2)
        L = 1.0   # pendulum length (m)
        
        # Time points
        t = np.linspace(0, 10, 100)
        
        # Generate various initial conditions
        initial_thetas = np.random.uniform(-np.pi/3, np.pi/3, self.samples)
        initial_omegas = np.random.uniform(-0.5, 0.5, self.samples)
        
        inputs = []
        targets = []
        
        for theta0, omega0 in zip(initial_thetas, initial_omegas):
            # Simplified pendulum motion (small angle approximation)
            theta = theta0 * np.cos(np.sqrt(g/L) * t) + omega0 * np.sin(np.sqrt(g/L) * t)
            omega = -theta0 * np.sqrt(g/L) * np.sin(np.sqrt(g/L) * t) + omega0 * np.cos(np.sqrt(g/L) * t)
            
            # Create input features: [t, theta(t), omega(t)]
            X = np.column_stack((t[:-1], theta[:-1], omega[:-1]))
            # Target is the next state: [theta(t+1), omega(t+1)]
            y = np.column_stack((theta[1:], omega[1:]))
            
            inputs.append(X)
            targets.append(y)
        
        self.data = torch.tensor(np.vstack(inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.vstack(targets), dtype=torch.float32)
    
    def save_dataset(self, save_path):
        """Save the dataset to a file.
        
        Args:
            save_path (str): Path to save the dataset.
        """
        if self.data is None or self.targets is None:
            raise ValueError("No data to save")
        
        np.savez(
            save_path,
            inputs=self.data.numpy(),
            targets=self.targets.numpy()
        )
        print(f"Dataset saved to {save_path}")


class WaveDataset(PhysicsDataset):
    """Dataset for wave propagation."""
    
    def __init__(self, data_path=None, transform=None, generate=False, 
                 grid_size=50, samples=100, save_path=None):
        """Initialize wave dataset.
        
        Args:
            data_path (str, optional): Path to the dataset file.
            transform (callable, optional): Transform to be applied to samples.
            generate (bool): Whether to generate synthetic data.
            grid_size (int): Size of the spatial grid for wave simulation.
            samples (int): Number of time steps to simulate.
            save_path (str, optional): Path to save generated data.
        """
        self.generate = generate
        self.grid_size = grid_size
        self.samples = samples
        self.save_path = save_path
        super().__init__(data_path, transform)
        
        if generate:
            self._generate_data()
            if save_path:
                self.save_dataset(save_path)
    
    def _load_data(self):
        """Load wave propagation data from file."""
        try:
            data = np.load(self.data_path)
            self.data = torch.tensor(data['inputs'], dtype=torch.float32)
            self.targets = torch.tensor(data['targets'], dtype=torch.float32)
        except Exception as e:
            raise IOError(f"Failed to load wave data: {e}")
    
    def _generate_data(self):
        """Generate synthetic wave propagation data using the wave equation."""
        # Initialize wave field with random initial conditions
        inputs = []
        targets = []
        
        # Generate multiple wave simulations with different initial conditions
        for _ in range(10):  # Generate 10 different wave scenarios
            # Create a grid
            x = np.linspace(0, 1, self.grid_size)
            y = np.linspace(0, 1, self.grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Initialize wave with Gaussian pulse
            center_x = np.random.uniform(0.3, 0.7)
            center_y = np.random.uniform(0.3, 0.7)
            sigma = np.random.uniform(0.05, 0.1)
            
            # Initial wave field
            u = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
            u_prev = u.copy()
            
            # Wave speed (can vary for different simulations)
            c = np.random.uniform(0.1, 0.3)
            
            # Spatial step
            dx = 1.0 / (self.grid_size - 1)
            
            # Time step (for stability)
            dt = 0.1 * dx / c
            
            # Generate time evolution
            for t in range(self.samples - 1):
                # Compute Laplacian using finite differences
                laplacian = (
                    np.roll(u, 1, axis=0) + 
                    np.roll(u, -1, axis=0) + 
                    np.roll(u, 1, axis=1) + 
                    np.roll(u, -1, axis=1) - 
                    4 * u
                ) / (dx**2)
                
                # Update wave field using wave equation: u_tt = c^2 * laplacian
                u_next = 2 * u - u_prev + c**2 * dt**2 * laplacian
                
                # Apply boundary conditions (set edges to zero)
                u_next[0, :] = u_next[-1, :] = u_next[:, 0] = u_next[:, -1] = 0
                
                # Store current state as input and next state as target
                inputs.append(u.flatten())
                targets.append(u_next.flatten())
                
                # Update for next iteration
                u_prev = u.copy()
                u = u_next.copy()
        
        # Convert to PyTorch tensors
        self.data = torch.tensor(np.array(inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)
    
    def save_dataset(self, save_path):
        """Save the dataset to a file."""
        if self.data is None or self.targets is None:
            raise ValueError("No data to save")
        
        np.savez(
            save_path,
            inputs=self.data.numpy(),
            targets=self.targets.numpy()
        )
        print(f"Dataset saved to {save_path}") 