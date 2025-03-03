"""Visualization utilities for physics-based AI."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


def plot_pendulum_trajectory(times, thetas, omegas, true_times=None, true_thetas=None, 
                            true_omegas=None, title="Pendulum Trajectory", save_path=None,
                            untrained_thetas=None, untrained_omegas=None,
                            random_thetas=None, random_omegas=None):
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
        untrained_thetas (array, optional): Predictions from untrained model.
        untrained_omegas (array, optional): Angular velocity predictions from untrained model.
        random_thetas (array, optional): Random predictions.
        random_omegas (array, optional): Random angular velocity predictions.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot angle
    ax1.plot(times, thetas, 'b-', label='Trained Model', linewidth=2)
    
    if true_times is not None and true_thetas is not None:
        ax1.plot(true_times, true_thetas, 'k-', label='Ground Truth', linewidth=2)
    
    if untrained_thetas is not None:
        ax1.plot(times, untrained_thetas, 'g--', label='Untrained Model', alpha=0.7)
    
    if random_thetas is not None:
        ax1.plot(times, random_thetas, 'r:', label='Random', alpha=0.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (rad)')
    ax1.set_title(f'{title} - Angle vs Time')
    ax1.grid(True)
    ax1.legend()
    
    # Plot angular velocity
    ax2.plot(times, omegas, 'b-', label='Trained Model', linewidth=2)
    
    if true_times is not None and true_omegas is not None:
        ax2.plot(true_times, true_omegas, 'k-', label='Ground Truth', linewidth=2)
    
    if untrained_omegas is not None:
        ax2.plot(times, untrained_omegas, 'g--', label='Untrained Model', alpha=0.7)
    
    if random_omegas is not None:
        ax2.plot(times, random_omegas, 'r:', label='Random', alpha=0.5)
    
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
                    true_thetas=None, untrained_thetas=None, random_thetas=None):
    """Create animation of pendulum motion.
    
    Args:
        times (array): Time points.
        thetas (array): Pendulum angles at each time point.
        L (float): Pendulum length in meters.
        title (str): Animation title.
        fps (int): Frames per second for the animation.
        save_path (str, optional): Path to save the animation. If None, the animation is displayed.
        true_thetas (array, optional): True pendulum angles for comparison.
        untrained_thetas (array, optional): Predictions from untrained model.
        random_thetas (array, optional): Random predictions.
    """
    # Count how many models we need to display
    model_count = 1  # Trained model
    if true_thetas is not None:
        model_count += 1
    if untrained_thetas is not None:
        model_count += 1
    if random_thetas is not None:
        model_count += 1
    
    # Set up the figure and grid for multiple pendulums
    fig = plt.figure(figsize=(model_count * 5, 6))
    
    # Create subplots for each model
    axes = []
    lines = []
    bobs = []
    
    # Determine number of columns and rows
    cols = min(model_count, 4)  # Maximum 4 columns
    rows = (model_count + cols - 1) // cols
    
    gs = GridSpec(rows, cols, figure=fig)
    
    # Add pendulums with appropriate titles and colors
    plot_idx = 0
    
    # Trained model (always present)
    ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
    ax.set_xlim([-1.2*L, 1.2*L])
    ax.set_ylim([-1.2*L, 1.2*L])
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Trained Model")
    line, = ax.plot([], [], 'b-', lw=2)  # Blue for trained model
    bob, = ax.plot([], [], 'bo', markersize=10)
    axes.append(ax)
    lines.append(line)
    bobs.append(bob)
    plot_idx += 1
    
    # Ground truth (if provided)
    if true_thetas is not None:
        ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
        ax.set_xlim([-1.2*L, 1.2*L])
        ax.set_ylim([-1.2*L, 1.2*L])
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Ground Truth")
        line, = ax.plot([], [], 'k-', lw=2)  # Black for ground truth
        bob, = ax.plot([], [], 'ko', markersize=10)
        axes.append(ax)
        lines.append(line)
        bobs.append(bob)
        plot_idx += 1
    
    # Untrained model (if provided)
    if untrained_thetas is not None:
        ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
        ax.set_xlim([-1.2*L, 1.2*L])
        ax.set_ylim([-1.2*L, 1.2*L])
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Untrained Model")
        line, = ax.plot([], [], 'g-', lw=2)  # Green for untrained
        bob, = ax.plot([], [], 'go', markersize=10)
        axes.append(ax)
        lines.append(line)
        bobs.append(bob)
        plot_idx += 1
    
    # Random model (if provided)
    if random_thetas is not None:
        ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
        ax.set_xlim([-1.2*L, 1.2*L])
        ax.set_ylim([-1.2*L, 1.2*L])
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Random Prediction")
        line, = ax.plot([], [], 'r-', lw=2)  # Red for random
        bob, = ax.plot([], [], 'ro', markersize=10)
        axes.append(ax)
        lines.append(line)
        bobs.append(bob)
    
    # Add a time display at the top
    time_text = fig.text(0.5, 0.95, '', ha='center')
    fig.suptitle(title, fontsize=16)
    
    def init():
        """Initialize animation."""
        for line, bob in zip(lines, bobs):
            line.set_data([], [])
            bob.set_data([], [])
        time_text.set_text('')
        return lines + bobs + [time_text]
    
    def update(i):
        """Update animation for frame i."""
        updates = []
        
        # Update trained model
        theta = thetas[i]
        x = L * np.sin(theta)
        y = -L * np.cos(theta)
        lines[0].set_data([0, x], [0, y])
        bobs[0].set_data([x], [y])
        updates.extend([lines[0], bobs[0]])
        
        # Update ground truth if provided
        model_idx = 1
        if true_thetas is not None:
            theta = true_thetas[i]
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            lines[model_idx].set_data([0, x], [0, y])
            bobs[model_idx].set_data([x], [y])
            updates.extend([lines[model_idx], bobs[model_idx]])
            model_idx += 1
        
        # Update untrained model if provided
        if untrained_thetas is not None:
            theta = untrained_thetas[i]
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            lines[model_idx].set_data([0, x], [0, y])
            bobs[model_idx].set_data([x], [y])
            updates.extend([lines[model_idx], bobs[model_idx]])
            model_idx += 1
        
        # Update random model if provided
        if random_thetas is not None:
            theta = random_thetas[i]
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            lines[model_idx].set_data([0, x], [0, y])
            bobs[model_idx].set_data([x], [y])
            updates.extend([lines[model_idx], bobs[model_idx]])
        
        time_text.set_text(f'Time: {times[i]:.2f}s')
        updates.append(time_text)
        
        return updates
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(times)),
                          init_func=init, blit=True, interval=1000/fps)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    return anim


def plot_wave_field(wave_field, grid_size=None, title="Wave Field", colorbar_label="Amplitude",
                   cmap='viridis', save_path=None, true_wave_field=None, 
                   untrained_wave_field=None, random_wave_field=None):
    """Plot 2D wave field.
    
    Args:
        wave_field (array): 2D array representing the wave field.
        grid_size (int, optional): Size of the grid if wave_field is flattened.
        title (str): Plot title.
        colorbar_label (str): Label for the colorbar.
        cmap (str): Matplotlib colormap name.
        save_path (str, optional): Path to save the figure. If None, the figure is displayed.
        true_wave_field (array, optional): True wave field for comparison.
        untrained_wave_field (array, optional): Prediction from untrained model.
        random_wave_field (array, optional): Random prediction.
    """
    # Reshape if needed
    if grid_size is not None and wave_field.ndim == 1:
        wave_field = wave_field.reshape(grid_size, grid_size)
    
    if true_wave_field is not None and grid_size is not None and true_wave_field.ndim == 1:
        true_wave_field = true_wave_field.reshape(grid_size, grid_size)
    
    if untrained_wave_field is not None and grid_size is not None and untrained_wave_field.ndim == 1:
        untrained_wave_field = untrained_wave_field.reshape(grid_size, grid_size)
    
    if random_wave_field is not None and grid_size is not None and random_wave_field.ndim == 1:
        random_wave_field = random_wave_field.reshape(grid_size, grid_size)
    
    # Count how many models we need to display
    model_count = 1  # Trained model
    if true_wave_field is not None:
        model_count += 1
    if untrained_wave_field is not None:
        model_count += 1
    if random_wave_field is not None:
        model_count += 1
    
    # Determine number of columns and rows
    cols = min(model_count, 2)  # Maximum 2 columns
    rows = (model_count + cols - 1) // cols
    
    fig = plt.figure(figsize=(cols * 6, rows * 5))
    
    # Add a main title
    fig.suptitle(title, fontsize=16)
    
    # Create subplots
    plot_idx = 1
    
    # Trained model
    ax = fig.add_subplot(rows, cols, plot_idx)
    im = ax.imshow(wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
    ax.set_title("Trained Model")
    plt.colorbar(im, ax=ax, label=colorbar_label)
    plot_idx += 1
    
    # Ground truth
    if true_wave_field is not None:
        ax = fig.add_subplot(rows, cols, plot_idx)
        im = ax.imshow(true_wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
        ax.set_title("Ground Truth")
        plt.colorbar(im, ax=ax, label=colorbar_label)
        plot_idx += 1
    
    # Untrained model
    if untrained_wave_field is not None:
        ax = fig.add_subplot(rows, cols, plot_idx)
        im = ax.imshow(untrained_wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
        ax.set_title("Untrained Model")
        plt.colorbar(im, ax=ax, label=colorbar_label)
        plot_idx += 1
    
    # Random predictions
    if random_wave_field is not None:
        ax = fig.add_subplot(rows, cols, plot_idx)
        im = ax.imshow(random_wave_field, cmap=cmap, origin='lower', interpolation='bilinear')
        ax.set_title("Random Prediction")
        plt.colorbar(im, ax=ax, label=colorbar_label)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title
    
    if save_path:
        plt.savefig(save_path)
        print(f"Wave field plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def animate_wave_field(wave_fields, grid_size=None, title="Wave Propagation", 
                      fps=10, cmap='viridis', save_path=None, true_wave_fields=None,
                      untrained_wave_fields=None, random_wave_fields=None):
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
        untrained_wave_fields (array, optional): Predictions from untrained model.
        random_wave_fields (array, optional): Random predictions.
    """
    # Reshape if needed
    if grid_size is not None and wave_fields.ndim == 2:
        num_frames = wave_fields.shape[0]
        wave_fields = wave_fields.reshape(num_frames, grid_size, grid_size)
    
    if true_wave_fields is not None and grid_size is not None and true_wave_fields.ndim == 2:
        num_frames = true_wave_fields.shape[0]
        true_wave_fields = true_wave_fields.reshape(num_frames, grid_size, grid_size)
    
    if untrained_wave_fields is not None and grid_size is not None and untrained_wave_fields.ndim == 2:
        num_frames = untrained_wave_fields.shape[0]
        untrained_wave_fields = untrained_wave_fields.reshape(num_frames, grid_size, grid_size)
    
    if random_wave_fields is not None and grid_size is not None and random_wave_fields.ndim == 2:
        num_frames = random_wave_fields.shape[0]
        random_wave_fields = random_wave_fields.reshape(num_frames, grid_size, grid_size)
    
    # Count how many models we need to display
    model_count = 1  # Trained model
    if true_wave_fields is not None:
        model_count += 1
    if untrained_wave_fields is not None:
        model_count += 1
    if random_wave_fields is not None:
        model_count += 1
    
    # Determine number of columns and rows
    cols = min(model_count, 2)  # Maximum 2 columns
    rows = (model_count + cols - 1) // cols
    
    # Get min and max values for consistent colormap across all models
    vmin = wave_fields.min()
    vmax = wave_fields.max()
    
    if true_wave_fields is not None:
        vmin = min(vmin, true_wave_fields.min())
        vmax = max(vmax, true_wave_fields.max())
    
    if untrained_wave_fields is not None:
        vmin = min(vmin, untrained_wave_fields.min())
        vmax = max(vmax, untrained_wave_fields.max())
    
    if random_wave_fields is not None:
        vmin = min(vmin, random_wave_fields.min())
        vmax = max(vmax, random_wave_fields.max())
    
    # Create figure and axes
    fig = plt.figure(figsize=(cols * 6, rows * 5))
    fig.suptitle(title, fontsize=16)
    
    # Create subplots
    axes = []
    images = []
    
    # Trained model
    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title("Trained Model")
    im = ax.imshow(wave_fields[0], cmap=cmap, origin='lower', 
                  interpolation='bilinear', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Amplitude')
    axes.append(ax)
    images.append(im)
    
    # Ground truth
    if true_wave_fields is not None:
        ax = fig.add_subplot(rows, cols, 2)
        ax.set_title("Ground Truth")
        im = ax.imshow(true_wave_fields[0], cmap=cmap, origin='lower', 
                      interpolation='bilinear', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Amplitude')
        axes.append(ax)
        images.append(im)
    
    # Untrained model
    if untrained_wave_fields is not None:
        ax = fig.add_subplot(rows, cols, 3)
        ax.set_title("Untrained Model")
        im = ax.imshow(untrained_wave_fields[0], cmap=cmap, origin='lower', 
                      interpolation='bilinear', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Amplitude')
        axes.append(ax)
        images.append(im)
    
    # Random predictions
    if random_wave_fields is not None:
        ax = fig.add_subplot(rows, cols, 4)
        ax.set_title("Random Prediction")
        im = ax.imshow(random_wave_fields[0], cmap=cmap, origin='lower', 
                      interpolation='bilinear', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Amplitude')
        axes.append(ax)
        images.append(im)
    
    # Add time display
    time_text = fig.text(0.5, 0.01, '', ha='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def update(i):
        """Update animation for frame i."""
        updates = []
        
        # Update trained model
        images[0].set_array(wave_fields[i])
        updates.append(images[0])
        
        # Update ground truth if provided
        model_idx = 1
        if true_wave_fields is not None:
            images[model_idx].set_array(true_wave_fields[i])
            updates.append(images[model_idx])
            model_idx += 1
        
        # Update untrained model if provided
        if untrained_wave_fields is not None:
            images[model_idx].set_array(untrained_wave_fields[i])
            updates.append(images[model_idx])
            model_idx += 1
        
        # Update random model if provided
        if random_wave_fields is not None:
            images[model_idx].set_array(random_wave_fields[i])
            updates.append(images[model_idx])
        
        time_text.set_text(f'Frame: {i}')
        updates.append(time_text)
        
        return updates
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(len(wave_fields)),
                          blit=True, interval=1000/fps)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
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


def generate_random_predictions(initial_value, num_steps, range_min=-np.pi, range_max=np.pi, smooth_factor=0.7):
    """Generate random predictions with some smoothness.
    
    Args:
        initial_value (float): Initial value.
        num_steps (int): Number of time steps to generate.
        range_min (float): Minimum value in the range.
        range_max (float): Maximum value in the range.
        smooth_factor (float): Smoothness factor between 0 and 1.
            - 0 means completely random
            - 1 means constant initial value
            
    Returns:
        array: Random predictions with some smoothness.
    """
    predictions = np.zeros(num_steps)
    predictions[0] = initial_value
    
    for i in range(1, num_steps):
        # Generate random value within range
        random_value = np.random.uniform(range_min, range_max)
        
        # Smooth with previous value
        predictions[i] = smooth_factor * predictions[i-1] + (1 - smooth_factor) * random_value
    
    return predictions 