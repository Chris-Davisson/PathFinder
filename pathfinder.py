import json
import math
import networkx as nx
import argparse
import time
import random

def load_graph(filename="example_graph.json"):
    """Loads a graph from a JSON file in node-link format."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges='links')

def select_landmarks(G, num_landmarks):
    """Selects the first 'num_landmarks' nodes as landmarks."""
    nodes = list(G.nodes())
    if num_landmarks > len(nodes):
        num_landmarks = len(nodes)
    return random.sample(nodes, num_landmarks)

def precompute_landmark_paths(G, landmarks):
    """Precomputes and stores shortest paths between landmarks."""
    landmark_paths = {}
    start_time = time.time()
    for i, landmark1 in enumerate(landmarks):
        # Check timer
        if time.time() - start_time > 60:
            print("Precompute timer: 60 seconds exceeded, returning partial results.")
            break
        # Limit the number of connections per landmark to sqrt(num_landmarks)
        num_connections = int(math.sqrt(len(landmarks)))
        # Connect to a subset of other landmarks
        for landmark2 in landmarks[i+1:i+1+num_connections]:
            # Check timer inside inner loop as well
            if time.time() - start_time > 60:
                print("Precompute timer: 60 seconds exceeded, returning partial results.")
                return landmark_paths
            try:
                path = nx.dijkstra_path(G, landmark1, landmark2, weight='weight')
                if landmark1 not in landmark_paths:
                    landmark_paths[landmark1] = {}
                landmark_paths[landmark1][landmark2] = path
            except nx.NetworkXNoPath:
                # No path between these landmarks
                pass
    return landmark_paths


def find_closest_landmark(G, node, landmarks):
    """Finds the landmark closest to a given node."""
    shortest_path_len = float('inf')
    closest_landmark = None
    path_to_landmark = []

    for landmark in landmarks:
        try:
            path = nx.dijkstra_path(G, node, landmark, weight='weight')
            path_len = len(path) - 1
            if path_len < shortest_path_len:
                shortest_path_len = path_len
                closest_landmark = landmark
                path_to_landmark = path
        except nx.NetworkXNoPath:
            continue
    return closest_landmark, path_to_landmark


def find_path_with_landmarks(G, source, dest, landmarks, landmark_paths):
    """Finds a path from source to destination using landmarks and returns the path and landmarks used."""
    # Find paths to the nearest landmarks for both source and destination
    source_landmark, path_to_source_landmark = find_closest_landmark(G, source, landmarks)
    dest_landmark, path_to_dest_landmark = find_closest_landmark(G, dest, landmarks)

    if source_landmark is None or dest_landmark is None:
        return None, None, None  # No path to a landmark

    # Check for direct path between the landmarks in our precomputed table
    # and handle both directions
    landmark_path = []
    if source_landmark in landmark_paths and dest_landmark in landmark_paths.get(source_landmark, {}):
        landmark_path = landmark_paths[source_landmark][dest_landmark]
    elif dest_landmark in landmark_paths and source_landmark in landmark_paths.get(dest_landmark, {}):
        # Reverse the path if found in the other direction
        landmark_path = list(reversed(landmark_paths[dest_landmark][source_landmark]))

    # If there is no precomputed path, try to compute one now
    if not landmark_path:
        try:
            landmark_path = nx.dijkstra_path(G, source_landmark, dest_landmark, weight='weight')
        except nx.NetworkXNoPath:
            return None, source_landmark, dest_landmark # No path between landmarks

    # Combine the paths
    full_path = path_to_source_landmark[:-1] + landmark_path + list(reversed(path_to_dest_landmark))[1:]

    return full_path, source_landmark, dest_landmark

def process_queries(G, landmarks, landmark_paths, query_file=False, output_file="paths_output.txt"):
    """Processes pathfinding queries and writes the full path and landmarks used to a file."""
    if not query_file:
        query_file = input("Enter query file path: ")

    with open(query_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            source, dest = map(int, line.strip().split())
            
            path, source_lm, dest_lm = find_path_with_landmarks(G, source, dest, landmarks, landmark_paths)

            f_out.write(f"Query: Find path from {source} to {dest}\n")
            if path:
                path_str = " -> ".join(map(str, path))
                f_out.write(f"  - Path: {path_str}\n")
                f_out.write(f"  - Landmarks Used: {source_lm} (from source), {dest_lm} (from destination)\n")
            else:
                f_out.write("  - No path found.\n")
            f_out.write("-" * 20 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pathfinder")
    parser.add_argument("-g", help="Path to JSON graph", default="example_graph.json")
    parser.add_argument("-q", default="queries.txt", help="Path to queries")
    args = parser.parse_args()
    # Load the graph
    G = load_graph(filename=args.g)
    num_nodes = len(G.nodes())
    print(f"Graph loaded with {num_nodes} nodes and {len(G.edges())} edges.")

    # Select landmarks
    num_landmarks = int(math.ceil(math.sqrt(num_nodes)))
    landmarks = select_landmarks(G, num_landmarks)
    print(f"Selected {len(landmarks)} landmarks.")

    # Precompute paths between landmarks
    print("Pre-computing paths between landmarks...")
    landmark_paths = precompute_landmark_paths(G, landmarks)
    print("Path pre-computation complete.")

    # Process queries and save to file
    print("Processing queries and saving paths to paths_output.txt...")
    process_queries(G, landmarks, landmark_paths, args.q)
    print("Done. Check paths_output.txt for results.")

if __name__ == "__main__":
    main()