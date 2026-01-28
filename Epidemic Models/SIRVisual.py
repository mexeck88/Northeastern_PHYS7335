#!/user/bin/env python3

""" SIRModel.py
Python module for simulating and visualizing the SIR model on networks.
This module generates various graph types and performs SIR epidemics on them.
It visualizes the spread of infection and compiles the frames into a GIF.

This module is a modified version of the RandomWalks.py module.

Author: Matt Eckert
Date: 28JAN2026
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

# Default Parameters
NUM_NODES = 100
BETA = 0.3    # Infection prob
GAMMA = 0.1   # Recovery prob
NUM_STEPS = 100
INITIAL_INFECTED = 1

USAGE = r"""
    SIRModel.py -n <Number of Nodes> -b <Infection Rate Beta> -g <Recovery Rate Gamma>

Flags (all optional; defaults shown in brackets):
    -n, --nodes <Number of Nodes>       Number of nodes [default: 100]
    -b, --beta <Infection Rate>         Infection probability (S->I) [default: 0.3]
    -g, --gamma <Recovery Rate>         Recovery probability (I->R) [default: 0.1]
    -s, --steps <Number of Steps>       Number of simulation steps [default: 100]
    -h, --help                          Show this help/usage message and exit
    """

def handle_sigint(_sig, _frame):
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

def sir_step(graph, states, beta, gamma):
    """
    Perform a single step of the SIR model.
    States: 'S', 'I', 'R'
    Returns: new_states (dict)
    """
    new_states = states.copy()
    nodes = list(graph.nodes())
    
    # Identify currently infected nodes to optimize
    infected_nodes = [n for n in nodes if states[n] == 'I']
    
    # Infection spread (S -> I)
    # Iterate over infected nodes and try to infect their susceptibility neighbors
    for node in infected_nodes:
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if states[neighbor] == 'S':
                # Attempt infection
                if random.random() < beta:
                    new_states[neighbor] = 'I'
    
    # Recovery (I -> R)
    for node in infected_nodes:
        # Note: A node can get infected and recover in the same step if we are not careful about order?
        # Standard discrete SIR usually updates simultaneously.
        # Here we use the OLD state to determine transitions, so a node that just became I in `new_states` 
        # (from S) is not in `infected_nodes` list, so it won't recover this turn. This is correct.
        
        if random.random() < gamma:
            new_states[node] = 'R'
            
    return new_states

def visualize_sir(graph, graph_name, beta, gamma, num_steps):
    """
    Perform SIR visualization for the passed graph.
    Generates frames and saves as GIF.
    """
    num_nodes = graph.number_of_nodes()
    # Layout
    # Layout: Increase k to spread out nodes and iterations for better convergence
    pos = nx.spring_layout(graph, k=3.0/np.sqrt(num_nodes), iterations=100)

    print(f"[INFO] Visualizing {graph_name} (Beta={beta}, Gamma={gamma})")

    # Create frames directory
    frames_dir = "frames_sir"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    else:
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
            
    # Initialize States
    states = {node: 'S' for node in graph.nodes()}
    
    # Pick random initial infected
    # Or use central node like RandomWalks.py?
    # RandomWalks uses central node. Let's use central node for consistency/visualization niceness.
    betweeness_centrality = nx.betweenness_centrality(graph)
    central_node = max(betweeness_centrality, key=betweeness_centrality.get)
    states[central_node] = 'I'
    print(f"[INFO] Initial patient zero: {central_node}")
    
    # Colors suitable for S, I, R
    # S: Blue (Susceptible/calm)
    # I: Red (Infected/danger)
    # R: Green (Recovered/safe)
    color_map = {'S': '#1f77b4', 'I': '#d62728', 'R': '#2ca02c'}
    
    # Initial Frame
    node_colors = [color_map[states[n]] for n in graph.nodes()]
    
    fig = plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
            node_size=600, width=2, font_size=10, font_weight='bold')
    plt.title(f"{graph_name} - SIR Model - Step 0", fontsize=14, fontweight='bold', pad=10)
    
    s_count = list(states.values()).count('S')
    i_count = list(states.values()).count('I')
    r_count = list(states.values()).count('R')
    stats_text = f"S: {s_count} | I: {i_count} | R: {r_count}\nBeta: {beta} | Mu: {gamma}"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, fontweight='bold')
    
    plt.axis('off')
    save_frame_minimal_whitespace(fig, os.path.join(frames_dir, "frame_0000.png"))
    
    # Simulation Loop
    for step in tqdm(range(num_steps), desc=f"Simulating SIR on {graph_name}"):
        states = sir_step(graph, states, beta, gamma)
        
        # Visualization
        node_colors = [color_map[states[n]] for n in graph.nodes()]
        
        fig = plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                node_size=600, width=2, font_size=10, font_weight='bold')
        
        plt.title(f"{graph_name} - SIR Model - Step {step + 1}", fontsize=14, fontweight='bold', pad=10)
        
        s_count = list(states.values()).count('S')
        i_count = list(states.values()).count('I')
        r_count = list(states.values()).count('R')
        stats_text = f"S: {s_count} | I: {i_count} | R: {r_count}\nBeta: {beta} | Mu: {gamma}"
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, fontweight='bold')
        
        plt.axis('off')
        save_frame_minimal_whitespace(fig, os.path.join(frames_dir, f"frame_{step + 1:04d}.png"))
        
        # Stop if no infected left
        if i_count == 0:
            print(f"[INFO] Epidemic ended at step {step + 1}")
            # Save a few more stagnant frames to pause the GIF at the end
            last_frame_path = os.path.join(frames_dir, f"frame_{step + 1:04d}.png")
            for extra in range(1, 6):
                shutil.copy(last_frame_path, os.path.join(frames_dir, f"frame_{step + 1 + extra:04d}.png"))
            break

    # Create GIF
    print("[INFO] Creating GIF...")
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        # Open, copy to memory, and close immediately to avoid file locks
        with Image.open(frame_path) as img:
            frames.append(img.copy())
        
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gif_path = os.path.join(output_dir, f"{graph_name}_SIR_animation.gif")
    # Duration per frame in ms
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                   duration=200, loop=0, optimize=False)
    print(f"[INFO] GIF saved as '{gif_path}'")
    
    os.system('rmdir /S /Q "{}"'.format(frames_dir))

    print("[INFO] Frames directory deleted.")

def parse_args():
    parser = argparse.ArgumentParser(description="SIR Model Visualization", usage=USAGE, add_help=False)
    parser.add_argument("-n", "--nodes", type=int, default=NUM_NODES, help="Number of nodes")
    parser.add_argument("-b", "--beta", type=float, default=BETA, help="Infection rate")
    parser.add_argument("-g", "--gamma", type=float, default=GAMMA, help="Recovery rate")
    parser.add_argument("-s", "--steps", type=int, default=NUM_STEPS, help="Number of steps")
    parser.add_argument("-h", "--help", action="store_true", help="Show help")
    
    args = parser.parse_args()
    if args.help:
        print(USAGE)
        sys.exit(0)
        
    return args.nodes, args.beta, args.gamma, args.steps

def main():
    num_nodes, beta, gamma, num_steps = parse_args()
    
    print(f"[INFO] Running SIR Model with N={num_nodes}, Beta={beta}, Gamma={gamma}")
    
    # Generate Graphs
    # 1. ER Graph
    er_graph = nx.erdos_renyi_graph(num_nodes, 0.5) # 0.5 probability for connectivity
    visualize_sir(er_graph, "ER_Graph", beta, gamma, num_steps)
    
    # # 2. WS Graph
    # ws_graph = nx.watts_strogatz_graph(num_nodes, k=4, p=0.1)
    # visualize_sir(ws_graph, "WS_Graph", beta, gamma, num_steps)
    
    # # 3. BA Graph
    # ba_graph = nx.barabasi_albert_graph(num_nodes, 2)
    # visualize_sir(ba_graph, "BA_Graph", beta, gamma, num_steps)
    
    # # 4. Configuration Model (using BA degrees)
    # ba_degrees = [d for n, d in ba_graph.degree()]
    # config_model = nx.configuration_model(ba_degrees)
    # # Config model makes MultiGraph, convert to Graph and remove self-loops for visualization clarity
    # config_model = nx.Graph(config_model)
    # config_model.remove_edges_from(nx.selfloop_edges(config_model))
    # visualize_sir(config_model, "Config_Model", beta, gamma, num_steps)
    
    print("[INFO] All simulations completed.")

if __name__ == "__main__":
    main()
