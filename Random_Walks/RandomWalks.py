#!/user/bin/env python3

""" RandomWalks.py
Python module for simulating random walks. Currently, this module generates a ER graph and then performs random walks on it.
Additionally, this code visualizes the random walks and compiles them into a GIF.

TODO:
- Add the three missing graph types (BA, WS, Config)
- Add the two missing deterministic approaches to Random Walk (Degree Block, Matrix)\
- Add graph functions (W_i vs t, W_i vs k, Mixing time)

Author: Matt Eckert

Date: 12JAN2026
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys  
import numpy as np
import signal
import argparse
from PIL import Image
import shutil
from tqdm import tqdm


NUM_NODES = 30
RATE_OF_ESCAPE = 0.80
NUM_WALKERS = 4
ER_PROBABILITY = 0.20
NUM_STEPS = 30
DISTRIBUTION_DENSITY = 50
USAGE = r"""
    RandomWalks.py -w <Number of Walkers for Stochastic Visualizations> -r <Rate of Escape for all analyses> -n <Number of Nodes for Stochastic Visualizations>

Flags (all optional; defaults shown in brackets):
    -w, --walkers <Number of Walkers for Stochastic Visualizations>    Number of walkers [default: 4]
    -r, --rate <Rate of Escape for all analyses>    Rate of escape [default: 0.80]
    -n, --nodes <Number of Nodes for Stochastic Visualizations>    Number of nodes [default: 30]
    -h, --help           Show this help/usage message and exit
    """



def handle_sigint(_sig, _frame):  # pylint: disable=invalid-name
    """
    Handle SIGINT signal
    """
    print("\nReceived SIGINT. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)


def save_frame_minimal_whitespace(fig, filename):
    """
    Save figure with minimal white space.
    """
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.01)
    plt.close()


def walker_step(graph, current_node, rate_of_escape):
    """
    Perform a single step for a walker.
    We will act from the walker outwards, each walker will attempt to walk on a randomly selected edge.
    """
    # Determine d_ij
    neighbors = list(graph.neighbors(current_node))

    if not neighbors:
        return current_node  # No movement possible

    # select a random neighbor
    next_node = random.choice(neighbors)
    d_ij = rate_of_escape / graph.degree[next_node] # movement probability

    if random.random() < d_ij:
        return next_node # move to next node
    else:
        return current_node # stay put

def parse_args():
    """
    Parse command line arguments
    """
    
    cli_parser = argparse.ArgumentParser(description="Random Walks Simulation", usage=USAGE, add_help=False)
    cli_parser.add_argument("-w", "--walkers",
                            type=int,
                            default=NUM_WALKERS,
                            help="Number of walkers")
    cli_parser.add_argument("-r", "--rate",
                            type=float,
                            default=RATE_OF_ESCAPE,
                            help="Rate of escape")
    cli_parser.add_argument("-n", "--nodes",
                            type=int,
                            default=NUM_NODES,
                            help="Number of nodes")
    cli_parser.add_argument("-h", "--help",
                            action="store_true",
                            help="Show this help message and exit")
    args = cli_parser.parse_args()
    
    if args.help:
        print(USAGE)
        sys.exit(0)

    # quick args check
    num_walkers = args.walkers
    rate_of_escape = args.rate
    num_nodes = args.nodes

    if num_walkers <= 0:
        raise ValueError("Number of walkers must be positive.")
    if not (0 < rate_of_escape < 1):
        raise ValueError("Rate of escape must be between 0 and 1.")
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive.")
    print(f"[INFO] Using {num_walkers} walkers, rate of escape {rate_of_escape}, {num_nodes} nodes")
    return num_walkers, rate_of_escape, num_nodes


def stochastic_visualization(graph, graph_name, num_walkers, rate_of_escape):
    """
    Perform stochastic random walk visualization for the passed graph. It will generate a 
    series of frames and save them to a directory called 'frames'. It will then create a 
    GIF from the frames and save it as '{graph_name}_random_walks_animation.gif'. Finally, it will 
    delete the frames directory.

    Args:
        graph: The graph to perform the random walk on.
        graph_name: The name of the graph to be used in the filename.
        num_walkers: The number of walkers to use.
        rate_of_escape: The rate of escape.

    Returns:
        None
    """
    num_nodes = graph.number_of_nodes()
    pos = nx.spring_layout(graph, k=1/np.sqrt(num_nodes) + 0.7*1/np.sqrt(num_nodes), iterations=25)
    betweeness_centrality = nx.betweenness_centrality(graph)
    central_node = max(betweeness_centrality, key=betweeness_centrality.get)

    print(f"[INFO] Central node determined to be: {central_node}")
    
    # Create frames directory
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else:
        # Clear existing frames
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
    
    # Highlight Central node red
    node_colors = ['red' if node == central_node else 'lightblue' for node in graph.nodes()]

    # Define walker colors (extremely vibrant colors using hsv colormap)
    walker_colors = plt.cm.hsv(np.linspace(0, 0.9, num_walkers))

    # Save the initial graph
    fig = plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
            node_size=600, width=2, font_size=10, font_weight='bold')
    plt.title(f"{graph_name} - Initial State", fontsize=14, fontweight='bold', pad=10)
    params_text = f"Walkers: {num_walkers} | Rate: {rate_of_escape} | Nodes: {num_nodes}"
    fig.text(0.5, 0.02, params_text, ha='center', fontsize=10, style='italic')
    plt.axis('off')
    save_frame_minimal_whitespace(fig, os.path.join(frames_dir, "frame_0000.png"))

    # Initialize walkers at the central node
    walkers = [central_node for _ in range(num_walkers)]
    walker_paths = [[central_node] for _ in range(num_walkers)]

    for step in tqdm(range(NUM_STEPS), desc="Simulating random walks"):
        # Execute a step for each walker
        for i in range(num_walkers):
            current_node = walkers[i]
            next_node = walker_step(graph, current_node, rate_of_escape)
            walkers[i] = next_node
            walker_paths[i].append(next_node)

        # Visualize current state
        fig = plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                node_size=600, width=2, font_size=10, font_weight='bold')
        
        # Draw walker paths and current positions
        for i, walker_node in enumerate(walkers):
            path = walker_paths[i]
            # Draw the path
            path_coords = np.array([pos[node] for node in path])
            plt.plot(path_coords[:, 0], path_coords[:, 1], 
                    color=walker_colors[i], alpha=0.7, linewidth=2.3, label=f'Walker {i+1}')
            
            # Draw current walker position as a larger marker
            walker_pos = pos[walker_node]
            plt.scatter(walker_pos[0], walker_pos[1], 
                       color=walker_colors[i], s=400, marker='o', 
                       edgecolors='black', linewidths=2, zorder=5)
            
            # Add walker number label
            plt.text(walker_pos[0], walker_pos[1], str(i+1), 
                    fontsize=11, fontweight='bold', ha='center', va='center',
                    color='white', zorder=6)
        
        plt.legend(loc='upper left', fontsize=9, framealpha=0.95)
        plt.title(f"Random Walk Simulation - Step {step + 1}/{NUM_STEPS}", fontsize=14, pad=10)
        params_text = f"Walkers: {num_walkers} | Rate: {rate_of_escape} | Nodes: {num_nodes}"
        fig.text(0.5, 0.02, params_text, ha='center', fontsize=10, style='italic')
        plt.axis('off')
        save_frame_minimal_whitespace(fig, os.path.join(frames_dir, f"frame_{step + 1:04d}.png"))
    
    # Create GIF from frames
    print("[INFO] Creating GIF from frames...")
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frames.append(Image.open(frame_path))
    
    # Save as GIF
    gif_path = os.path.join("output", f"{graph_name}_random_walks_animation.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                   duration=450, loop=0, optimize=False)
    print(f"[INFO] GIF saved as '{gif_path}'")
    
    # Delete frames directory
    shutil.rmtree(frames_dir)
    print("[INFO] Frames directory deleted.")

def walkers_by_degree_plot(walkers_by_degree_list, approach_name):
    """
    Plots the number of walkers on node i vs the degree of node i.
    Stationary state of the graphs. Will plot the curve for all four graphs on one plot.
    Will save the plot as a png file in the output directory.

    Args:
        walkers_by_degree_list: list of walkers by degree distributions for each graph, 
        where each element is a the list of tuples of (walker_count, degree)
    
    """
    labels = ["ER", "WS", "BA", "Config"]
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    plt.figure(figsize=(10, 8))
    
    for i, dist in enumerate(walkers_by_degree_list):
        # dist is a list of tuples (walker_count, degree)
        # Aggregate by degree
        degree_counts = {}
        for count, degree in dist:
            if degree not in degree_counts:
                degree_counts[degree] = []
            degree_counts[degree].append(count)
        
        # Calculate mean walkers for each degree
        unique_degrees = sorted(degree_counts.keys())
        mean_walkers = [np.mean(degree_counts[d]) for d in unique_degrees]
        
        print(f"[DEBUG] Plotting {labels[i]}: {len(unique_degrees)} unique degrees. Range: {min(unique_degrees)}-{max(unique_degrees)} Walker Range: {min(mean_walkers):.2f}-{max(mean_walkers):.2f}")
        
        plt.scatter(unique_degrees, mean_walkers, label=labels[i], color=colors[i], marker=markers[i], alpha=0.7)
        # Optional: Plot line to see trends better if desired
        # plt.plot(unique_degrees, mean_walkers, color=colors[i], alpha=0.3)

    plt.title(f"{approach_name} - Walkers vs Degree (Stationary State)")
    plt.xlabel("Degree (Log)")
    plt.ylabel("Average Number of Walkers (Log)")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.savefig(os.path.join("output", f"{approach_name}_walkers_by_degree.png"))
    plt.close()
    print(f"[INFO] Saved output/{approach_name}_walkers_by_degree.png")

def walkers_by_time_plot(walkers_by_time_list, approach_name):
    """
    Plots the number of walkers on node i vs the time step.
    Will plot the curve for all four graphs on one plot.
    Will save the plot as a png file in the output directory.

    Args:
        walkers_by_time_list: list of walkers by time distributions for each graph
        approach_name: string name of the approach (e.g., "Stochastic" or "Block")
    """
    time_steps = range(len(walkers_by_time_list[0]))
    labels = ["ER", "WS", "BA", "Config"]
    colors = ['blue', 'green', 'red', 'purple']

    plt.figure(figsize=(10, 8))
    
    for i, dist in enumerate(walkers_by_time_list):
        plt.plot(time_steps, dist, label=labels[i], color=colors[i], alpha=0.7)

    plt.title(f"Cumulative walkers vs Time ({approach_name} Analysis)")
    plt.xlabel("Time Step")
    plt.ylabel("Total Accumulated Walkers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("output", f"{approach_name}_cumulative_walkers_by_time.png"))
    plt.close()
    print(f"[INFO] Saved output/{approach_name}_cumulative_walkers_by_time.png")

def mixing_time_plot(mixing_time_list):
    """
    Plots the mixing time (counts vs time) for varying r values for each graph.
    Generates a single figure with four subplots, one for each graph type.

    Args:
        mixing_time_list: list of 2D mixing time array for each graph, 
        the 2D array is of shape (r value, time steps) with content being the number of walkers at that time step
    """
    # Recalculate r_values to match stochastic_analysis
    r_values = np.linspace(0.1, 1, 5)
    graph_labels = ["ER", "WS", "BA", "Config"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, mixing_matrix in enumerate(mixing_time_list):
        ax = axes[i]
        
        # mixing_matrix is shape (5, time_steps)
        # Plot a curve for each r value
        time_steps = range(len(mixing_matrix[0]))
        
        for r_idx, r in enumerate(r_values):
            ax.plot(time_steps, mixing_matrix[r_idx], label=f"r={r:.2f}", alpha=0.7)

        ax.set_title(f"Mixing Time Analysis - {graph_labels[i]}")
        ax.set_xlabel("Time Step (Log)")
        ax.set_ylabel("Number of Walkers (Log)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(title="Escape Rate", fontsize='small')
        ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    filename = "mixing_time_combined.png"
    plt.savefig(os.path.join("output", filename))
    plt.close()
    print(f"[INFO] Saved {filename}")


def stochastic_analysis(graph, num_walkers, rate_of_escape):
    """
    Performs the stochastic analysis of the graphs, This still uses the smaller stochastic graphs used for the visualizations.
    It will produce the following distributions for each graph:
    - walkers by degree
    - walkers by time
    - mixing time
    
    Args:
    - graph: graph to perform stochastic analysis on
    - num_walkers: number of walkers to use
    - rate_of_escape: rate of escape for walkers

    Returns:
    - walkers_by_degree: walkers by degree distribution
    - walkers_by_time: walkers by time distribution
    - mixing_time: mixing time
    """
    
    walkers_by_degree = []
    walkers_by_time = []
    
    stationary_state_time = 100

    mixing_time_array = np.zeros((5, stationary_state_time))
    num_nodes = len(graph.nodes())
    betweeness_centrality = nx.betweenness_centrality(graph)
    central_node = max(betweeness_centrality, key=betweeness_centrality.get)

    walkers = [central_node for _ in range(num_walkers)]
    walker_paths = [[central_node] for _ in range(num_walkers)]

    # Select a random node for W_i vs time
    # Select a random node for W_i vs time
    random_node = random.choice(list(graph.nodes()))
    walkers_by_time = [0]*stationary_state_time
    
    cumulative_walkers = 0
    for step in tqdm(range(stationary_state_time), desc="Simulating random walks (For time and degree analysis)"):
        # Execute a step for each walker
        for i in range(num_walkers):
            current_node = walkers[i]
            next_node = walker_step(graph, current_node, rate_of_escape)
            walkers[i] = next_node
            walker_paths[i].append(next_node)

        # Count the number of walkers on the random node
        cumulative_walkers += walkers.count(random_node)
        walkers_by_time[step] = cumulative_walkers

    # Count the number of walkers on each node for W_i vs degree
    walkers_by_degree = [0]*num_nodes
    for node in graph.nodes():
        # check the number of walkers on each node
        walkers_by_degree[node] = (walkers.count(node), graph.degree[node])
    
    # Mixing time
    # need to run the sim for varying r values, and produce a w vs t plot
    # will do 5 r values
    r_values = np.linspace(0.1, 1, 5)
    for r_idx, r in tqdm(enumerate(r_values), total=len(r_values), desc="Varying r values for mixing time analysis"):
        # Reset walkers for each r value
        walkers = [central_node for _ in range(num_walkers)]
        
        for step in tqdm(range(stationary_state_time), desc="Simulating random walks (For mixing time analysis)", leave=False):
            # Execute a step for each walker
            for i in range(num_walkers):
                current_node = walkers[i]
                next_node = walker_step(graph, current_node, r)
                walkers[i] = next_node
            
            # Count the number of walkers on the random node
            mixing_time_array[r_idx][step] = walkers.count(random_node)
        
    return walkers_by_degree, walkers_by_time, mixing_time_array


def degree_block_analysis(graph, num_walkers, rate_of_escape):
    """
    Performs the degree block analysis of the graphs.
    """
    
    
    stationary_state_time = 1000
    mixing_time_array = np.zeros((5, stationary_state_time))
    
    # Pre-calculate graph properties
    # Pre-calculate graph properties
    num_nodes = graph.number_of_nodes()
    
    # Ensure degrees match the node order in nodes_list/arrays
    degree_dict = dict(graph.degree())
    nodes_list = list(graph.nodes())
    degrees = np.array([degree_dict[n] for n in nodes_list])
    avg_degree = np.mean(degrees)
    
    # Identify central node (Max Degree is sufficient and much faster than betweenness)
    # betweenness = nx.betweenness_centrality(graph) 
    # central_node = max(betweenness, key=betweenness.get)
    central_node = max(degree_dict, key=degree_dict.get)

    # Convert nodes to integer indices if they aren't already, for array indexing
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    central_node_idx = node_to_idx[central_node]
    
    random_node = random.choice(nodes_list)
    random_node_idx = node_to_idx[random_node]
    
    W_current = np.zeros(num_nodes)
    W_current[central_node_idx] = num_walkers
    
    walkers_by_time = np.zeros(stationary_state_time)
    injection_base = (degrees / avg_degree) * (num_walkers / num_nodes)
    
    # Debug print
    if num_nodes > 100: # Only for deterministic
        print(f"[DEBUG] Node Analysis: Avg Deg={avg_degree:.2f}, Central={central_node} (Deg={degree_dict[central_node]})")
    
    cumulative_walkers = 0
    for t in tqdm(range(stationary_state_time), desc="Degree Block Analysis"):
        # Record stats (Cumulative Sum)
        cumulative_walkers += W_current[random_node_idx]
        walkers_by_time[t] = cumulative_walkers 
        
        # Update Equation: W_k(t+1) = W_k(t) * (1-r) + r * injection_base
        W_next = W_current * (1 - rate_of_escape) + (rate_of_escape * injection_base)
        W_current = W_next

    walkers_by_degree = []
    for i in range(num_nodes):
        walkers_by_degree.append((W_current[i], degrees[i]))

    r_values = np.linspace(0.1, 1, 5)
    
    for r_idx, r_val in tqdm(enumerate(r_values), total=len(r_values), desc="Degree Block Mixing Analysis"):
        # Reset system for each r
        W_mix = np.zeros(num_nodes)
        W_mix[central_node_idx] = num_walkers
        
        for t in range(stationary_state_time):
            mixing_time_array[r_idx][t] = W_mix[random_node_idx]
            
            # Update with current r_val
            W_mix = W_mix * (1 - r_val) + (r_val * injection_base)

    return walkers_by_degree, walkers_by_time, mixing_time_array
    

def matrix_analysis(graph, num_walkers, rate_of_escape):
    """
    Performs the matrix analysis of the graphs.
    """
    # TODO

def main():
    """
    Main function to perform our three random walk approaches on the four graph types.
    """
    num_walkers, rate_of_escape, num_nodes = parse_args()

    # Generate the 8 graphs needed (smaller for stochastic, larger for deterministic)
    stoch_er_graph = nx.erdos_renyi_graph(num_nodes, 0.2)
    stoch_ws_graph = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
    stoch_ba_graph = nx.barabasi_albert_graph(num_nodes, 2)
    # Use BA degree sequence for Configuration Model
    ba_degrees = [d for n, d in stoch_ba_graph.degree()]
    stoch_config_model = nx.configuration_model(ba_degrees)

    # Generate the deterministic graphs
    det_num_nodes = 1000
    det_walkers = 10000
    det_rate_of_escape = 0.2

    det_er_graph = nx.erdos_renyi_graph(det_num_nodes, 0.2)
    det_ws_graph = nx.watts_strogatz_graph(det_num_nodes, 4, 0.1)
    det_ba_graph = nx.barabasi_albert_graph(det_num_nodes, 2)
    
    # Use BA degree sequence for Deterministic Configuration Model
    det_ba_degrees = [d for n, d in det_ba_graph.degree()]
    det_config_model = nx.configuration_model(det_ba_degrees)

    # Distributions for plots
    time_dist = np.linspace(0, NUM_STEPS, DISTRIBUTION_DENSITY)
    degree_dist = np.linspace(0, num_nodes, DISTRIBUTION_DENSITY)
    escape_rate_dist = np.linspace(0, rate_of_escape, DISTRIBUTION_DENSITY)


    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the stochastic visualizations
    print("[INFO] Running stochastic visualizations...")
    stochastic_visualization(stoch_er_graph, "ER_Graph", num_walkers, rate_of_escape)
    stochastic_visualization(stoch_ws_graph, "WS_Graph", num_walkers, rate_of_escape)
    stochastic_visualization(stoch_ba_graph, "BA_Graph", num_walkers, rate_of_escape)
    stochastic_visualization(stoch_config_model, "Config_Model", num_walkers, rate_of_escape)

    print("[INFO] Running Stochastic Analysis...")
    degree_dists_stochastic = []
    time_dists_stochastic = []
    mixing_matrices_stochastic = []

    # ER
    deg, time, mix = stochastic_analysis(stoch_er_graph, num_walkers, rate_of_escape)
    degree_dists_stochastic.append(deg)
    time_dists_stochastic.append(time)
    mixing_matrices_stochastic.append(mix)

    # WS
    deg, time, mix = stochastic_analysis(stoch_ws_graph, num_walkers, rate_of_escape)
    degree_dists_stochastic.append(deg)
    time_dists_stochastic.append(time)
    mixing_matrices_stochastic.append(mix)

    # BA
    deg, time, mix = stochastic_analysis(stoch_ba_graph, num_walkers, rate_of_escape)
    degree_dists_stochastic.append(deg)
    time_dists_stochastic.append(time)
    mixing_matrices_stochastic.append(mix)

    # Config
    deg, time, mix = stochastic_analysis(stoch_config_model, num_walkers, rate_of_escape)
    degree_dists_stochastic.append(deg)
    time_dists_stochastic.append(time)
    mixing_matrices_stochastic.append(mix)

    # Plotting
    print("[INFO] Generating plots...")
    # walkers_by_degree_plot(degree_dists_stochastic, "Stochastic")
    walkers_by_time_plot(time_dists_stochastic, "Stochastic")
    mixing_time_plot(mixing_matrices_stochastic)


    print("[INFO] Running Deterministic Analysis...")
    degree_dists_block = []
    time_dists_block = []
    mixing_matrices_block = []

    # ER
    deg, time, mix = degree_block_analysis(det_er_graph, det_walkers, det_rate_of_escape)
    degree_dists_block.append(deg)
    time_dists_block.append(time)
    mixing_matrices_block.append(mix)

    # WS
    deg, time, mix = degree_block_analysis(det_ws_graph, det_walkers, det_rate_of_escape)
    degree_dists_block.append(deg)
    time_dists_block.append(time)
    mixing_matrices_block.append(mix)

    # BA
    deg, time, mix = degree_block_analysis(det_ba_graph, det_walkers, det_rate_of_escape)
    degree_dists_block.append(deg)
    time_dists_block.append(time)
    mixing_matrices_block.append(mix)

    # Config
    deg, time, mix = degree_block_analysis(det_config_model, det_walkers, det_rate_of_escape)
    degree_dists_block.append(deg)
    time_dists_block.append(time)
    mixing_matrices_block.append(mix)

    # Plotting
    print("[INFO] Generating plots...")
    walkers_by_degree_plot(degree_dists_block, "Block")
    # walkers_by_degree_plot(degree_dists_matrix, "Matrix")
    mixing_time_plot(mixing_matrices_block)
    walkers_by_time_plot(time_dists_block, "Block")
    # mixing_matrices_matrix = []

    # Plotting
    print("[INFO] Generating plots...")
    walkers_by_degree_plot(degree_dists_block, "Block")
    # walkers_by_degree_plot(degree_dists_matrix, "Matrix")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    main()
