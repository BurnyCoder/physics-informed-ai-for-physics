"""Visualization utilities for physics-based AI."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


def plot_pendulum_trajectory(times, thetas, omegas, true_times=None, true_thetas=None, 
                            true_omegas=None, title="Pendulum Trajectory", save_path=None):
    """Plot pendulum trajectory.
    
    Args:
        times (array): Time points.
        thetas (array): Pendulum angles at each time point.
        omegas (array): Angular velocities at each time point.
        true_times (array, optional): True time points for comparison.
        true_thetas (array, optional): True pendulum angles for comparison.
        true_omegas (array, optional): True angular velocities for comparison.
        title (str): Plot title.
        save_path (str, optional): Path to save the figure. If None, the figure is displayed.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot angle
    ax1.plot(times, thetas, 'b-', label='Predicted')
    if true_times is not None and true_thetas is not None:
        ax1.plot(true_times, true_thetas, 'r--', label='True')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (rad)')
    ax1.set_title(f'{title} - Angle vs Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot angular velocity
    ax2.plot(times, omegas, 'b-', label='Predicted')
    if true_times is not None and true_omegas is not None:
        ax2.plot(true_times, true_omegas, 'r--', label='True')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Angular Velocity vs Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()


def animate_pendulum(times, thetas, L=1.0, title="Pendulum Animation", fps=30, save_path=None,
                    true_thetas=None):
    """Create animation of pendulum motion.
    
    Args:
        times (array): Time points.
        thetas (array): Pendulum angles at each time point.
        L (float): Pendulum length in meters.
        title (str): Animation title.
        fps (int): Frames per second for the animation.
        save_path (str, optional): Path to save the animation. If None, the animation is displayed.
        true_thetas (array, optional): True pendulum angles for comparison.
    """
    # Set up the figure
    fig = plt.figure(figsize=(12, 6))
    
    if true_thetas is not None:
        # Side-by-side comparison
        gs = GridSpec(1, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        ax1.set_xlim([-1.2*L, 1.2*L])
        ax1.set_ylim([-1.2*L, 1.2*L])
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.set_title("Predicted Motion")
        
        ax2.set_xlim([-1.2*L, 1.2*L])
        ax2.set_ylim([-1.2*L, 1.2*L])
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.set_title("True Motion")
        
        # Pendulum components - predicted
        line1, = ax1.plot([], [], 'k-', lw=2)
        bob1, = ax1.plot([], [], 'bo', markersize=10)
        
        # Pendulum components - true
        line2, = ax2.plot([], [], 'k-', lw=2)
        bob2, = ax2.plot([], [], 'ro', markersize=10)
        
        time_text = fig.text(0.5, 0.95, '', ha='center')
        
        fig.suptitle(title, fontsize=16)
        
        def init():
            """Initialize animation."""
            line1.set_data([], [])
            bob1.set_data([], [])
            line2.set_data([], [])
            bob2.set_data([], [])
            time_text.set_text('')
            return line1, bob1, line2, bob2, time_text
        
        def update(i):
            """Update animation for frame i."""
            # Predicted pendulum
            theta = thetas[i]
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            
            line1.set_data([0, x], [0, y])
            bob1.set_data([x], [y])
            
            # True pendulum
            true_theta = true_thetas[i]
            true_x = L * np.sin(true_theta)
            true_y = -L * np.cos(true_theta)
            
            line2.set_data([0, true_x], [0, true_y])
            bob2.set_data([true_x], [true_y])
            
            time_text.set_text(f'Time: {times[i]:.2f}s')
            
            return line1, bob1, line2, bob2, time_text
    
    else:
        # Single pendulum
        ax = fig.add_subplot(111)
        ax.set_xlim([-1.2*L, 1.2*L])
        ax.set_ylim([-1.2*L, 1.2*L])
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)
        
        # Pendulum components
        line, = ax.plot([], [], 'k-', lw=2)
        bob, = ax.plot([], [], 'bo', markersize=10)
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
        
        def init():
            """Initialize animation."""
            line.set_data([], [])
            bob.set_data([], [])
            time_text.set_text('')
            return line, bob, time_text
        
        def update(i):
            """Update animation for frame i."""
            theta = thetas[i]
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            
            line.set_data([0, x], [0, y])
            bob.set_data([x], [y])
            time_text.set_text(f'Time: {times[i]:.2f}s')
            return line, bob, time_text
    
    # Create animation
    if true_thetas is not None:
        anim = FuncAnimation(fig, update, frames=range(len(times)),
                              init_func=init, blit=True, interval=1000/fps)
    else:
        anim = FuncAnimation(fig, update, frames=range(len(times)),
                              init_func=init, blit=True, interval=1000/fps)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    return anim


def plot_wave_field(wave_field, grid_size=None, title="Wave Field", colorbar_label="Amplitude",
                   cmap='viridis', save_path=None, true_wave_field=None):
    """Plot 2D wave field.
    
    Args:
        wave_field (array): 2D array representing the wave field.
        grid_size (int, optional): Size of the grid if wave_field is flattened.
        title (str): Plot title.
        colorbar_label (str): Label for the colorbar.
        cmap (str): Matplotlib colormap name.
        save_path (str, optional): Path to save the figure. If None, the figure is displayed.
        true_wave_field (array, optional): True wave field for comparison.
    """
    # Reshape if needed
    if grid_size is not None and wave_field.ndim == 1:
        wave_field = wave_field.reshape(grid_size, grid_size)
    
    if true_wave_field is not None and grid_size is not None and true_wave_field.ndim == 1:
        true_wave_field = true_wave_field.reshape(grid_size, grid_size)
    
    if true_wave_field is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot predicted wave field
        im1 = ax1.imshow(wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
        ax1.set_title(f"{title} - Predicted")
        fig.colorbar(im1, ax=ax1, label=colorbar_label)
        
        # Plot true wave field
        im2 = ax2.imshow(true_wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
        ax2.set_title(f"{title} - True")
        fig.colorbar(im2, ax=ax2, label=colorbar_label)
        
        # Add a main title
        fig.suptitle(title, fontsize=16)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=colorbar_label)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Wave field plot saved to {save_path}")
    else:
        plt.show()


def animate_wave_field(wave_fields, grid_size=None, title="Wave Propagation", 
                      fps=10, cmap='viridis', save_path=None, true_wave_fields=None):
    """Create animation of wave propagation.
    
    Args:
        wave_fields (array): 3D array of shape (num_frames, height, width) or 
                            2D array of shape (num_frames, flattened_grid).
        grid_size (int, optional): Size of the grid if wave_fields are flattened.
        title (str): Animation title.
        fps (int): Frames per second for the animation.
        cmap (str): Matplotlib colormap name.
        save_path (str, optional): Path to save the animation. If None, the animation is displayed.
        true_wave_fields (array, optional): True wave fields for comparison.
    """
    # Reshape if needed
    if grid_size is not None and wave_fields.ndim == 2:
        num_frames = wave_fields.shape[0]
        wave_fields = wave_fields.reshape(num_frames, grid_size, grid_size)
    
    if true_wave_fields is not None and grid_size is not None and true_wave_fields.ndim == 2:
        num_frames = true_wave_fields.shape[0]
        true_wave_fields = true_wave_fields.reshape(num_frames, grid_size, grid_size)
    
    # Get min and max values for consistent colormap
    if true_wave_fields is not None:
        vmin = min(wave_fields.min(), true_wave_fields.min())
        vmax = max(wave_fields.max(), true_wave_fields.max())
    else:
        vmin = wave_fields.min()
        vmax = wave_fields.max()
    
    if true_wave_fields is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.set_title("Predicted")
        ax2.set_title("True")
        
        # Initial frame
        im1 = ax1.imshow(wave_fields[0], cmap=cmap, origin='lower', 
                        interpolation='bilinear', vmin=vmin, vmax=vmax)
        im2 = ax2.imshow(true_wave_fields[0], cmap=cmap, origin='lower', 
                        interpolation='bilinear', vmin=vmin, vmax=vmax)
        
        fig.colorbar(im1, ax=ax1, label='Amplitude')
        fig.colorbar(im2, ax=ax2, label='Amplitude')
        
        time_text = fig.text(0.5, 0.95, '', ha='center')
        fig.suptitle(title, fontsize=16)
        
        def update(i):
            """Update animation for frame i."""
            im1.set_array(wave_fields[i])
            im2.set_array(true_wave_fields[i])
            time_text.set_text(f'Frame: {i}')
            return [im1, im2, time_text]
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title)
        
        # Initial frame
        im = ax.imshow(wave_fields[0], cmap=cmap, origin='lower', 
                      interpolation='bilinear', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white')
        
        def update(i):
            """Update animation for frame i."""
            im.set_array(wave_fields[i])
            time_text.set_text(f'Frame: {i}')
            return [im, time_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(wave_fields)),
                          blit=True, interval=1000/fps)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    return anim


def create_comparison_directory(base_dir):
    """Create a directory for saving comparison visualizations.
    
    Args:
        base_dir (str): Base directory path.
        
    Returns:
        str: Path to the created directory.
    """
    comp_dir = os.path.join(base_dir, 'comparisons')
    os.makedirs(comp_dir, exist_ok=True)
    return comp_dir 