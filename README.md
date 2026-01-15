# Northeastern_PHYS7335
Collection of projects and assignments from Northeastern University's Dynamical Processes on Complex Networks course.

## Random Walks

### Installation

1. Clone the whole repo, or download just the Random_walks directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Basic Usage
Run with default parameters:
```bash
python RandomWalks.py
```

#### Custom Parameters
```bash
python RandomWalks.py --walkers 6 --rate 0.75 --nodes 50 --probability 0.15
```

#### Command-Line Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--walkers` | `-w` | 4 | Number of walkers in the simulation |
| `--rate` | `-r` | 0.85 | Rate of escape (movement probability, 0-1) |
| `--nodes` | `-n` | 30 | Number of nodes in the graph |
| `--probability` | `-p` | 0.20 | Erdős-Rényi edge probability (0-1) |
| `--help` | `-h` | - | Display help message |

### Default Parameters

- **Number of Walkers**: 4
- **Rate of Escape**: 0.85
- **Number of Nodes**: 30
- **ER Probability**: 0.20
- **Simulation Steps**: 30
