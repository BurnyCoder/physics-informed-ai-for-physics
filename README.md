# Physics-Based AI for Physics

This project implements deep learning models specifically designed for physics problems. The repository contains tools for:

- Loading and preprocessing physics datasets
- Building physics-informed neural networks (PINNs)
- Training models with physical constraints
- Evaluating model performance
- Visualizing results

## Examples

### Pendulum Simulation

The pendulum model demonstrates how physics-informed neural networks can predict the motion of a simple pendulum by incorporating physical laws into the learning process.

![Pendulum Simulation Example](example.gif)

### Wave Propagation

The wave propagation model shows how our approach can simulate complex wave dynamics while respecting the underlying wave equation.

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `data/`: Contains dataset handling code and example datasets
- `models/`: Neural network architectures for physics problems
- `training/`: Training utilities, loss functions, and physics constraints
- `utils/`: Helper functions for visualization and evaluation
- `examples/`: Example scripts for training and inference

## Usage

See the examples directory for detailed usage examples:

```bash
# Run a pendulum training example
python examples/train_pendulum.py --num_epochs 50 --physics_weight 0.7

# Run a wave propagation example
python examples/train_wave.py --grid_size 32 --num_epochs 50
```

### Command Line Parameters

#### Pendulum Model (`train_pendulum.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_path` | str | None | Path to existing dataset (if not specified, data will be generated) |
| `--save_data_path` | str | 'data/datasets/pendulum_data.npz' | Path to save generated dataset |
| `--checkpoint_dir` | str | 'checkpoints' | Directory to save model checkpoints |
| `--log_dir` | str | 'logs' | Directory for TensorBoard logs |
| `--hidden_dim` | int | 64 | Dimension of hidden layers in the model |
| `--num_layers` | int | 4 | Number of hidden layers in the model |
| `--batch_size` | int | 32 | Batch size for training |
| `--num_epochs` | int | 50 | Number of training epochs |
| `--learning_rate` | float | 1e-3 | Learning rate for optimizer |
| `--physics_weight` | float | 0.7 | Weight for physics-informed loss (between 0 and 1) |
| `--samples` | int | 1000 | Number of samples to generate if creating data |
| `--val_split` | float | 0.2 | Fraction of data to use for validation |
| `--no_cuda` | flag | False | Disable CUDA training |
| `--save_freq` | int | 10 | Frequency of saving checkpoints (epochs) |

#### Wave Model (`train_wave.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_path` | str | None | Path to existing dataset (if not specified, data will be generated) |
| `--save_data_path` | str | 'data/datasets/wave_data.npz' | Path to save generated dataset |
| `--checkpoint_dir` | str | 'checkpoints/wave' | Directory to save model checkpoints |
| `--log_dir` | str | 'logs/wave' | Directory for TensorBoard logs |
| `--grid_size` | int | 32 | Size of the spatial grid (grid_size x grid_size) |
| `--wave_speed` | float | 0.2 | Wave propagation speed |
| `--samples` | int | 100 | Number of time steps to simulate per scenario |
| `--hidden_dim` | int | 128 | Dimension of hidden layers in the model |
| `--num_layers` | int | 6 | Number of hidden layers in the model |
| `--batch_size` | int | 16 | Batch size for training |
| `--num_epochs` | int | 50 | Number of training epochs |
| `--learning_rate` | float | 1e-3 | Learning rate for optimizer |
| `--physics_weight` | float | 0.7 | Weight for physics-informed loss (between 0 and 1) |
| `--val_split` | float | 0.2 | Fraction of data to use for validation |
| `--no_cuda` | flag | False | Disable CUDA training |
| `--save_freq` | int | 10 | Frequency of saving checkpoints (epochs) |

## Supported Physics Problems

- Pendulum motion dynamics
- Wave propagation
- Fluid dynamics (basic) 

## Todo

- Hyperparameter tuning
- More complex physics problems
- More complex models
- More complex datasets
- More complex evaluation
- More complex visualization
