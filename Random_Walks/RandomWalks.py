#!/user/bin/env python3

""" RandomWalks.py
Python module for simulating random walks. Currently, this module generates a ER graph and then performs random walks on it.
Additionally, this code visualizes the random walks and compiles them into a GIF.

Author: Matt Eckert

Date: 12JAN2026
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys  
import numpy as np
import signal
import argparse
from PIL import Image
import shutil
from tqdm import tqdm


# Constant parameters (could be input args later)
NUM_NODES = 30
RATE_OF_ESCAPE = 0.85
NUM_WALKERS = 4
ER_PROBABILITY = 0.20
NUM_STEPS = 30
USAGE = r"""
    RandomWalks.py -w <Number of Walkers> -r <Rate of Escape> -n <Number of Nodes> -p <ER Probability>

Flags (all optional; defaults shown in brackets):
    -w, --walkers <Number of Walkers>    Number of walkers [default: 4]
    -r, --rate <Rate of Escape>    Rate of escape [default: 0.85]
    -n, --nodes <Number of Nodes>    Number of nodes [default: 30]
    -p, --probability <ER Probability>    ER graph probability [default: 0.20]
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




def main():
    # Generate a ER graph
    # Generate a set of walkers
    # Determine "central" node
    # Perform the random walks, each walker path will be a different color
    # Walkers will attempt to walk towards each edge node
    # save each step as an image
    # compile images into a gif

    # Parse command line arguments
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
    cli_parser.add_argument("-p", "--probability",
                            type=float,
                            default=ER_PROBABILITY,
                            help="ER graph probability")
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
    er_probability = args.probability

    if num_walkers <= 0:
        raise ValueError("Number of walkers must be positive.")
    if not (0 < rate_of_escape < 1):
        raise ValueError("Rate of escape must be between 0 and 1.")
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive.")
    if not (0 <= er_probability <= 1):
        raise ValueError("ER probability must be between 0 and 1.")
    print(f"[INFO] Using {num_walkers} walkers, rate of escape {rate_of_escape}, {num_nodes} nodes, ER probability {er_probability}")

    graph = nx.erdos_renyi_graph(num_nodes, er_probability)
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
    plt.title("Erdős-Rényi Graph - Initial State", fontsize=14, fontweight='bold', pad=10)
    params_text = f"Walkers: {num_walkers} | Rate: {rate_of_escape} | Nodes: {num_nodes} | ER Prob: {er_probability}"
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
        params_text = f"Walkers: {num_walkers} | Rate: {rate_of_escape} | Nodes: {num_nodes} | ER Prob: {er_probability}"
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
    gif_path = "random_walks_animation.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                   duration=450, loop=0, optimize=False)
    print(f"[INFO] GIF saved as '{gif_path}'")
    
    # Delete frames directory
    shutil.rmtree(frames_dir)
    print(f"[INFO] Frames directory deleted.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    main()
