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

# --- Global Configuration & Timers ---

PROGRAM_START = time.monotonic()
GLOBAL_DEADLINE_SECONDS = 60.0
SIZE_THRESHOLD_BYTES = 300 * 1024 * 1024  # 300 MB
PER_QUERY_TIMEOUT_SECONDS = 3.0

def deadline_reached():
    """Checks if the global preprocessing time limit has been exceeded."""
    return (time.monotonic() - PROGRAM_START) >= GLOBAL_DEADLINE_SECONDS

# --- Core Pathfinding Algorithm ---

def bounded_dijkstra(graph, start_node, end_node=None, targets=None, max_expansions=20000, time_budget=1.5):
    """
    Performs a Dijkstra search with limits on node expansions and execution time.

    Args:
        graph (nx.Graph): The graph to search.
        start_node: The starting node for the search.
        end_node: A single target node. If found, the search terminates early.
        targets (set): A set of target nodes. If any are found, the search terminates.
        max_expansions (int): The maximum number of nodes to pop from the priority queue.
        time_budget (float): The maximum wall-clock time in seconds for the search.

    Returns:
        A tuple (path, cost, reason), where:
        - path (list): The list of nodes from start to the found target, or None.
        - cost (float): The sum of edge weights along the path, or float('inf').
        - reason (str): A string indicating the outcome ('found', 'timeout', 'unreachable').
    """
    if start_node not in graph:
        return None, float('inf'), f"start_node_missing"

    search_start_time = time.monotonic()
    
    pq = [(0, start_node)]  # (cost, node)
    distances = {start_node: 0}
    parents = {start_node: None}
    expansions = 0

    target_set = set()
    if end_node:
        target_set.add(end_node)
    if targets:
        target_set.update(targets)

    while pq and expansions < max_expansions:
        if time.monotonic() - search_start_time > time_budget:
            return None, float('inf'), "timeout"

        cost, current_node = heapq.heappop(pq)
        expansions += 1

        if cost > distances.get(current_node, float('inf')):
            continue

        if current_node in target_set:
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            return path[::-1], cost, "found"

        for neighbor, edge_data in graph[current_node].items():
            weight = float(edge_data.get('weight', 1.0))
            new_cost = cost + weight
            if new_cost < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_cost
                parents[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))

    if end_node and end_node not in distances:
        return None, float('inf'), "unreachable"
        
    return None, float('inf'), "expansion_limit"


# --- Graph Loading ---

def load_graph_streaming(path):
    """
    Loads a graph from a large JSON file using a streaming parser (ijson).
    
    This function periodically checks the global deadline and will return a
    partially loaded graph if the deadline is exceeded.

    Args:
        path (str): The path to the node-link JSON graph file.

    Returns:
        nx.Graph: The loaded graph, which may be partial if the deadline was hit.
    """
    print(f"INFO: File is large. Using streaming loader for {path}...", file=sys.stderr)
    graph = nx.Graph()
    current_object_type = None
    node_count, link_count = 0, 0

    with open(path, 'rb') as f:
        try:
            parser = ijson.parse(f)
            for prefix, event, value in parser:
                if (node_count + link_count) % 10000 == 0 and deadline_reached():
                    print(f"WARNING: Global deadline reached during streaming. Loaded {node_count} nodes and {link_count} links.", file=sys.stderr)
                    return graph

                if prefix == 'nodes' and event == 'start_array':
                    current_object_type = 'nodes'
                elif prefix == 'links' and event == 'start_array':
                    current_object_type = 'links'
                elif '.id' in prefix and current_object_type == 'nodes':
                    graph.add_node(value)
                    node_count += 1
                elif prefix.endswith('.source'):
                    source = value
                elif prefix.endswith('.target'):
                    target = value
                elif prefix.endswith('.weight'):
                    weight = float(value)
                    graph.add_edge(source, target, weight=weight)
                    link_count += 1
                elif event == 'end_map' and current_object_type == 'links' and 'weight' not in graph[source][target]:
                    graph.add_edge(source, target, weight=1.0)
                    link_count += 1
        except ijson.JSONError as e:
            print(f"ERROR: Failed to parse JSON stream: {e}", file=sys.stderr)

    print(f"INFO: Streaming load complete. Loaded {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.", file=sys.stderr)
    return graph

def load_graph(path):
    """
    Loads a graph from a JSON file, choosing a method based on file size.

    Args:
        path (str): The path to the node-link JSON graph file.

    Returns:
        nx.Graph: The loaded graph. Returns an empty graph if deadline hit before loading.
    """
    if deadline_reached():
        print("WARNING: Deadline reached before graph loading could start.", file=sys.stderr)
        return nx.Graph()
        
    file_size = os.path.getsize(path)
    if file_size >= SIZE_THRESHOLD_BYTES:
        return load_graph_streaming(path)
    else:
        print(f"INFO: File is small. Using standard JSON loader for {path}...", file=sys.stderr)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if deadline_reached():
                print("WARNING: Deadline reached after standard load but before graph creation. Returning empty graph.", file=sys.stderr)
                return nx.Graph()
            
            # Manually build graph to ensure weight defaults
            G = nx.Graph()
            for node in data.get('nodes', []):
                G.add_node(node['id'])
            for link in data.get('links', []):
                G.add_edge(link['source'], link['target'], weight=float(link.get('weight', 1.0)))

            return G
        except (json.JSONDecodeError, KeyError) as e:
            print(f"ERROR: Failed to load or parse small JSON file: {e}", file=sys.stderr)
            return nx.Graph()


# --- Landmark Preprocessing ---

def select_landmarks(graph, k):
    """
    Selects k landmarks using degree-biased random sampling.

    Nodes with higher degrees have a higher probability of being chosen.

    Args:
        graph (nx.Graph): The graph.
        k (int): The number of landmarks to select.

    Returns:
        list: A list of k selected landmark node IDs.
    """
    if not graph or graph.number_of_nodes() == 0:
        return []

    nodes = list(graph.nodes())
    degrees = [d for n, d in graph.degree(nodes)]
    total_degree = sum(degrees)

    if total_degree == 0:  # Graph with nodes but no edges
        return random.sample(nodes, min(k, len(nodes)))

    probabilities = [d / total_degree for d in degrees]
    
    # Use random.choices for sampling with replacement, then unique-ify
    chosen = set()
    while len(chosen) < k and len(chosen) < len(nodes):
        selected_nodes = random.choices(nodes, weights=probabilities, k=k - len(chosen))
        chosen.update(selected_nodes)

    return list(chosen)

def precompute_landmark_paths(graph, landmarks):
    """
    Precomputes shortest paths between a subset of landmark pairs.

    Stops immediately if the global deadline is reached.

    Args:
        graph (nx.Graph): The graph.
        landmarks (list): A list of landmark node IDs.

    Returns:
        dict: A dictionary mapping (lm1, lm2) tuples to a (path, cost) tuple.
    """
    if not landmarks:
        return {}

    print("INFO: Starting landmark path precomputation...", file=sys.stderr)
    precomputed_paths = {}
    k = len(landmarks)
    subset_size = math.ceil(math.sqrt(k))
    
    start_time = time.monotonic()
    
    for i, lm1 in enumerate(landmarks):
        if deadline_reached():
            print("WARNING: Deadline reached during precomputation. Stopping.", file=sys.stderr)
            break
        
        # Connect to a subset of subsequent landmarks in the list
        for j in range(1, subset_size + 1):
            lm2 = landmarks[(i + j) % k]
            if lm1 == lm2 or (lm1, lm2) in precomputed_paths or (lm2, lm1) in precomputed_paths:
                continue

            # Use a very small budget for precomputation searches
            path, cost, reason = bounded_dijkstra(graph, lm1, lm2, max_expansions=10000, time_budget=0.2)
            
            if reason == 'found':
                precomputed_paths[(lm1, lm2)] = (path, cost)
                precomputed_paths[(lm2, lm1)] = (path[::-1], cost)

    end_time = time.monotonic()
    print(f"INFO: Precomputation finished in {end_time - start_time:.2f}s. Found {len(precomputed_paths)//2} inter-landmark paths.", file=sys.stderr)
    return precomputed_paths

# --- Query Processing ---

def find_nearest_landmark(graph, node, landmarks):
    """
    Finds the closest landmark to a given node using a bounded search.

    Args:
        graph (nx.Graph): The graph.
        node: The node ID to search from.
        landmarks (list): The list of landmark nodes.

    Returns:
        A tuple (path_to_landmark, cost_to_landmark, landmark_node). Returns
        (None, inf, None) if no landmark is reachable within bounds.
    """
    if not landmarks:
        return None, float('inf'), None

    max_exp = max(20000, 4 * math.ceil(math.sqrt(graph.number_of_edges() or 1)))
    
    path, cost, reason = bounded_dijkstra(
        graph,
        start_node=node,
        targets=set(landmarks),
        max_expansions=max_exp,
        time_budget=1.5
    )

    if path:
        return path, cost, path[-1]
    
    return None, float('inf'), None

def stitch_path(p1, p2, p3):
    """
    Stitches three path segments together, removing overlapping nodes.

    Args:
        p1 (list): First path segment (e.g., source to source_lm).
        p2 (list): Second path segment (e.g., source_lm to dest_lm).
        p3 (list): Third path segment (e.g., dest_lm to dest).

    Returns:
        list: The combined path.
    """
    if not p1 or not p2 or not p3:
        return None
    # p1 = [s..lm1], p2 = [lm1..lm2], p3 = [lm2..d]
    # result = p1 + p2[1:] + p3[1:]
    full_path = p1
    if p2[0] == full_path[-1]:
        full_path.extend(p2[1:])
    else: # Should not happen if logic is correct
        return None 
    
    if p3[0] == full_path[-1]:
        full_path.extend(p3[1:])
    else: # Should not happen
        return None
        
    return full_path

def get_path_cost(graph, path):
    """Calculates the total weight of a path."""
    if not path or len(path) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        cost += float(graph[u][v].get('weight', 1.0))
    return cost
    
def query_path(graph, landmarks, landmark_paths, source, dest):
    """
    Handles a single path query using the landmark routing strategy.

    Args:
        graph (nx.Graph): The graph.
        landmarks (list): List of landmark nodes.
        landmark_paths (dict): Precomputed paths between landmarks.
        source: The source node ID.
        dest: The destination node ID.

    Returns:
        A dictionary containing the query result details.
    """
    start_time = time.monotonic()
    
    # Phase 1: Find nearest landmarks
    path_to_slm, cost_to_slm, source_lm = find_nearest_landmark(graph, source, landmarks)
    path_to_dlm, cost_to_dlm, dest_lm = find_nearest_landmark(graph, dest, landmarks)

    if not source_lm or not dest_lm:
        result = {
            "cost": float('inf'), "phase": "no_path", "path": [],
            "source_lm": source_lm, "dest_lm": dest_lm,
            "reason": "Could not reach a landmark from source or destination."
        }
        result["time"] = time.monotonic() - start_time
        return result

    # Phase 2: Find path between landmarks
    inter_lm_path, inter_lm_cost = None, float('inf')
    
    if source_lm == dest_lm:
        # If nearest landmark is the same, the path is source -> lm -> dest
        inter_lm_path = [source_lm]
        inter_lm_cost = 0.0
    elif (source_lm, dest_lm) in landmark_paths:
        inter_lm_path, inter_lm_cost = landmark_paths[(source_lm, dest_lm)]
    else:
        # Fallback to on-demand bounded search
        path, cost, reason = bounded_dijkstra(graph, source_lm, dest_lm, time_budget=1.0)
        if reason == 'found':
            inter_lm_path, inter_lm_cost = path, cost

    # Phase 3: Stitch and finalize
    if inter_lm_path:
        # Reverse the path from destination to its landmark
        path_from_dlm = path_to_dlm[::-1]
        
        full_path = stitch_path(path_to_slm, inter_lm_path, path_from_dlm)
        
        if full_path:
            total_cost = get_path_cost(graph, full_path)
            # Sanity check costs
            # total_cost = cost_to_slm + inter_lm_cost + cost_to_dlm
            result = {
                "cost": total_cost, "phase": "stitched", "path": full_path,
                "source_lm": source_lm, "dest_lm": dest_lm
            }
        else:
            result = {
                "cost": float('inf'), "phase": "no_path", "path": [],
                "source_lm": source_lm, "dest_lm": dest_lm, "reason": "Path stitching failed."
            }
    else:
        result = {
            "cost": float('inf'), "phase": "no_path", "path": [],
            "source_lm": source_lm, "dest_lm": dest_lm, "reason": "No path found between landmarks."
        }
        
    result["time"] = time.monotonic() - start_time
    return result


def process_queries(graph, landmarks, landmark_paths, queries_path, output_file):
    """
    Reads queries from a file, processes them, and writes results to output.

    Args:
        graph (nx.Graph): The graph.
        landmarks (list): List of landmark nodes.
        landmark_paths (dict): Precomputed paths between landmarks.
        queries_path (str): Path to the file containing queries.
        output_file: A file-like object (e.g., sys.stdout or opened file) to write to.
    """
    print("INFO: Processing queries...", file=sys.stderr)
    query_count = 0
    with open(queries_path, 'r') as f:
        for line in f:
            query_count += 1
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            source, dest = parts[0], parts[1]
            
            # Attempt to cast to int if they look like ints, else treat as strings
            try:
                source = int(source)
            except ValueError:
                pass
            try:
                dest = int(dest)
            except ValueError:
                pass


            if source not in graph or dest not in graph:
                output_file.write(
                    f"Query {source} -> {dest}\n"
                    f"  - Path length (cost): inf\n"
                    f"  - Search time (s): 0.000\n"
                    f"  - Phase: no_path\n"
                    f"  - Reason: Source or destination node not in graph.\n"
                    f"  - Landmarks Used: None (from source), None (from destination)\n\n"
                )
                continue

            result = query_path(graph, landmarks, landmark_paths, source, dest)
            
            output_file.write(f"Query {source} -> {dest}\n")
            output_file.write(f"  - Path length (cost): {result['cost']:.4f}\n")
            output_file.write(f"  - Search time (s): {result['time']:.4f}\n")
            output_file.write(f"  - Phase: {result['phase']}\n")
            if result['path']:
                output_file.write(f"  - Path: {' -> '.join(map(str, result['path']))}\n")
            else:
                 output_file.write(f"  - Reason: {result.get('reason', 'N/A')}\n")
            output_file.write(f"  - Landmarks Used: {result['source_lm']} (from source), {result['dest_lm']} (from destination)\n\n")
            output_file.flush()
    print(f"INFO: Processed {query_count} queries.", file=sys.stderr)


# --- Main Execution ---

def main():
    """Main function to parse arguments and run the pathfinder."""
    parser = argparse.ArgumentParser(description="Large-Graph Pathfinder with Landmark Routing")
    parser.add_argument("-g", "--graph", required=True, help="Path to the node-link JSON graph file")
    parser.add_argument("-q", "--queries", required=True, help="Path to the queries file (each line: 'source dest')")
    parser.add_argument("-o", "--output", default="-", help="Output file path ('-' for stdout)")
    args = parser.parse_args()

    # --- Preprocessing ---
    print(f"--- PREPROCESSING (Max {GLOBAL_DEADLINE_SECONDS}s) ---", file=sys.stderr)
    
    graph = load_graph(args.graph)
    if not graph:
        print("ERROR: Graph is empty after loading. Cannot proceed.", file=sys.stderr)
        return

    print(f"INFO: Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.", file=sys.stderr)
    print(f"INFO: Time elapsed: {time.monotonic() - PROGRAM_START:.2f}s", file=sys.stderr)
    
    landmarks = []
    landmark_paths = {}

    if not deadline_reached():
        k = math.ceil(math.sqrt(graph.number_of_nodes()))
        landmarks = select_landmarks(graph, k)
        print(f"INFO: Selected {len(landmarks)} landmarks.", file=sys.stderr)
        print(f"INFO: Time elapsed: {time.monotonic() - PROGRAM_START:.2f}s", file=sys.stderr)
    else:
        print("WARNING: Deadline reached after graph loading, skipping landmark selection.", file=sys.stderr)

    if not deadline_reached() and landmarks:
        landmark_paths = precompute_landmark_paths(graph, landmarks)
        print(f"INFO: Time elapsed at end of preprocessing: {time.monotonic() - PROGRAM_START:.2f}s", file=sys.stderr)
    else:
         print("WARNING: Deadline reached, skipping landmark precomputation.", file=sys.stderr)

    print("\n--- QUERYING ---", file=sys.stderr)

    # --- Querying ---
    if args.output == "-":
        process_queries(graph, landmarks, landmark_paths, args.queries, sys.stdout)
    else:
        with open(args.output, 'w') as f_out:
            process_queries(graph, landmarks, landmark_paths, args.queries, f_out)
            
    print("\n--- Done ---", file=sys.stderr)


if __name__ == "__main__":
    main()