"""Visualization utilities for physics-based AI."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def animate_pendulum(times, thetas, L=1.0, title="Pendulum Animation", fps=30, save_path=None):
    """Create animation of pendulum motion.
    
    Args:
        times (array): Time points.
        thetas (array): Pendulum angles at each time point.
        L (float): Pendulum length in meters.
        title (str): Animation title.
        fps (int): Frames per second for the animation.
        save_path (str, optional): Path to save the animation. If None, the animation is displayed.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([-1.2*L, 1.2*L])
    ax.set_ylim([-1.2*L, 1.2*L])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    
    # Pendulum components
    line, = ax.plot([], [], 'k-', lw=2)  # Pendulum rod
    bob, = ax.plot([], [], 'bo', markersize=10)  # Pendulum bob
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
        bob.set_data([x], [y])  # Fix: Provide x and y as sequences
        time_text.set_text(f'Time: {times[i]:.2f}s')
        return line, bob, time_text
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(times)),
                           init_func=init, blit=True, interval=1000/fps)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    return anim


def plot_wave_field(wave_field, grid_size=None, title="Wave Field", colorbar_label="Amplitude",
                   cmap='viridis', save_path=None):
    """Plot 2D wave field.
    
    Args:
        wave_field (array): 2D array representing the wave field.
        grid_size (int, optional): Size of the grid if wave_field is flattened.
        title (str): Plot title.
        colorbar_label (str): Label for the colorbar.
        cmap (str): Matplotlib colormap name.
        save_path (str, optional): Path to save the figure. If None, the figure is displayed.
    """
    # Reshape if needed
    if grid_size is not None and wave_field.ndim == 1:
        wave_field = wave_field.reshape(grid_size, grid_size)
    
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
                      fps=10, cmap='viridis', save_path=None):
    """Create animation of wave propagation.
    
    Args:
        wave_fields (array): 3D array of shape (num_frames, height, width) or 
                            2D array of shape (num_frames, flattened_grid).
        grid_size (int, optional): Size of the grid if wave_fields are flattened.
        title (str): Animation title.
        fps (int): Frames per second for the animation.
        cmap (str): Matplotlib colormap name.
        save_path (str, optional): Path to save the animation. If None, the animation is displayed.
    """
    # Reshape if needed
    if grid_size is not None and wave_fields.ndim == 2:
        num_frames = wave_fields.shape[0]
        wave_fields = wave_fields.reshape(num_frames, grid_size, grid_size)
    
    # Get min and max values for consistent colormap
    vmin = wave_fields.min()
    vmax = wave_fields.max()
    
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