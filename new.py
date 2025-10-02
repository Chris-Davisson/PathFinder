import argparse
import time
import math
import random
import heapq
import sys
import os
import json
from collections import defaultdict
import networkx as nx
import ijson
import signal

# --- Global Configuration & Timers ---

PROGRAM_START = time.monotonic()
GLOBAL_DEADLINE_SECONDS = 60.0
SIZE_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB
# PER_QUERY_TIMEOUT_SECONDS = 3.0
SMALL_GRAPH_NODE_THRESHOLD = 10000000
IS_SMALL_GRAPH = None  # To be determined after loading the graph

# --- Define a custom exception for the timeout ---
class TimeoutInterrupt(Exception):
    """Custom exception for deadline interrupts."""
    pass

# --- Define the signal handler ---
def deadline_handler(signum, frame):
    """This function is called when the alarm signal is received."""
    print("INTERRUPT: Global 60-second deadline reached. Stopping preprocessing.", file=sys.stderr)
    raise TimeoutInterrupt

def get_query_pairs(query_file):
    """Generator to yield (source, dest) pairs from the query file."""
    with open(query_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                source = int(parts[0])
                dest = int(parts[1])
                yield (source, dest)
            except ValueError:
                continue

def process_small_graph_queries(G, queries_path, output_path):
    """Process queries for small graphs using full Dijkstra."""
    queries = list(get_query_pairs(queries_path))
    print(f"Processing {len(queries)} queries on small graph using full Dijkstra.", file=sys.stderr)
    for source, dest in queries:
        try:
            path = nx.bidirectional_dijkstra(G, source, dest, weight='weight')
            output_line = (
                f"SRC: {str(source).ljust(8)} | "
                f"DEST: {str(dest).ljust(8)} | "
                f"PATH: [{', '.join(map(str, path)).ljust(40)}]"
            )
        except nx.NetworkXNoPath:
            output_line = f"No path from {source} to {dest}"
        except nx.NodeNotFound as e:
            output_line = f"Node not found: {e}"
        if output_path == "-":
            print(output_line)
        else:
            with open(output_path, 'a') as out_f:
                out_f.write(output_line + "\n")


def main():
    """Main function to parse arguments and run the pathfinder."""
    parser = argparse.ArgumentParser(description="Large-Graph Pathfinder with Landmark Routing")
    parser.add_argument("-g", "--graph", required=True, help="Path to the node-link JSON graph file")
    parser.add_argument("-q", "--queries", required=True, help="Path to the queries file (each line: 'source dest')")
    parser.add_argument("-o", "--output", default="-", help="Output file path ('-' for stdout)")
    args = parser.parse_args()

    try:
        signal.signal(signal.SIGALRM, deadline_handler)
        signal.alarm(int(GLOBAL_DEADLINE_SECONDS))
    except Exception as e:
        print("Error setting up timer, program will not work on windows")

    print(f"--- PREPROCESSING (Max {GLOBAL_DEADLINE_SECONDS}s) ---", file=sys.stderr)
    G = None
    landmarks = []
    landmark_paths = {}

    try:
        with open(args.graph, 'r') as f:
            data = json.load(f)
        G = nx.node_link_graph(data, edges = "links")
        graph_size_bytes = os.path.getsize(args.graph)
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\nGraph Size: {graph_size_bytes}", file=sys.stderr)
        # Small graphs
        if graph_size_bytes <= SIZE_THRESHOLD_BYTES:
            print("Small graph detected, using full Dijkstra preprocessing.", file=sys.stderr)
            IS_SMALL_GRAPH = True
        else:
            print("Large graph detected, using landmark-based preprocessing.", file=sys.stderr)
            print(f"Graph size (bytes): {graph_size_bytes}", file=sys.stderr)
            IS_SMALL_GRAPH = False
    except TimeoutInterrupt:
        print("Preprocessing interrupted due to timeout.", file=sys.stderr)
        if G is None:
            print("No graph loaded, exiting.", file=sys.stderr)
            return
    finally:
        signal.alarm(0)  # Disable the alarm
        print(f"Preprocessing completed in {time.monotonic() - PROGRAM_START:.2f} seconds.", file=sys.stderr)


        
    if IS_SMALL_GRAPH:
        process_small_graph_queries(G, args.queries, args.output)
    else:
        print("Landmark-based pathfinding not yet implemented.", file=sys.stderr)
        # Placeholder for landmark-based pathfinding logic
        pass


if __name__ == "__main__":
    main()