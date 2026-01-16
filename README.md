# Northeastern_PHYS7335
Collection of projects and assignments from Northeastern University's Dynamical Processes on Complex Networks course.

## Random Walks

Simulation and visualization of random walks on various graph topologies (Erdős-Rényi, Watts-Strogatz, Barabási-Albert, and Configuration Model).

### Features

- **Stochastic Visualization**: Generates animated GIFs of random walkers on small graphs.
- **Stochastic Analysis**: Analyzes walker distribution and mixing times on small graphs.
- **Deterministic Analysis**: Performs "Degree Block" analysis on larger graphs (1000 nodes).
- **Graph Types**: Supports ER, WS, BA, and Configuration Model graphs.

### Installation

1. Clone the repo or download the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Basic Usage
Run with default parameters (Walkers: 4, Rate: 0.80, Nodes: 30):
```bash
python RandomWalks.py
```

#### Custom Parameters
Customize the visualization parameters:
```bash
python RandomWalks.py --walkers 6 --rate 0.75 --nodes 50
```

#### Command-Line Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--walkers` | `-w` | 4 | Number of walkers (Stochastic Visualization) |
| `--rate` | `-r` | 0.80 | Rate of escape (0-1) for all analyses |
| `--nodes` | `-n` | 30 | Number of nodes (Stochastic Visualization) |
| `--help` | `-h` | - | Display help message |

#### Output

The script creates an `output/` directory containing:
- **GIFs**: Animations of the random walks (e.g., `ER_Graph_random_walks_animation.gif`).
- **Plots**:
    - `*_walkers_by_degree.png`: Walkers vs Degree.
    - `*_cumulative_walkers_by_time.png`: Cumulative walkers over time.
    - `mixing_time_combined.png`: Mixing time analysis.

#### Notes

- The **Stochastic Visualization** uses the parameters provided via command line.
- The **Deterministic Analysis** runs on hardcoded larger graphs (1000 nodes, 10000 walkers) to ensure statistical significance.
