import argparse
import time
import math
import random
import heapq
import sys
import os
import json
import networkx as nx
import signal
import psutil
import ijson

# --- Global Configuration & Timers ---

PROGRAM_START = time.monotonic()
GLOBAL_DEADLINE_SECONDS = 60.0
# A 100MB JSON file is a much safer threshold to avoid memory expansion issues.
SIZE_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB
IS_SMALL_GRAPH = None
LIMIT = 1 << 30  # 1 GiB


# --- Custom Exception for Timeout ---
class TimeoutInterrupt(Exception):
    """Custom exception for deadline interrupts."""
    pass


def limit():
    """Enforce 1 GiB process memory cap (macOS safe)."""
    rss = psutil.Process(os.getpid()).memory_info().rss
    if rss > LIMIT:
        # Added a more descriptive error message.
        sys.stderr.write(f"\nERROR: Memory limit exceeded ({rss / (1<<30):.2f} GiB > {LIMIT / (1<<30):.2f} GiB).\n")
        sys.stderr.write("This can happen when the in-memory graph is much larger than its on-disk file.\n")
        sys.exit(137)

# --- Signal Handler ---
def deadline_handler(signum, frame):
    """This function is called when the alarm signal is received."""
    print("INTERRUPT: Global deadline reached. Stopping preprocessing.", file=sys.stderr)
    raise TimeoutInterrupt

# --- Graph Loading ---

def load_graph_streaming(path):
    """Loads a large graph using a streaming parser to save memory."""
    print("INFO: Using streaming loader for large graph...", file=sys.stderr)
    G = nx.Graph()
    with open(path, 'rb') as f:
        # We only need to parse the links, as add_edge will create nodes automatically.
        # This is the most memory-intensive part of the file.
        links = ijson.items(f, 'links.item')
        for link in links:
            source = link['source']
            target = link['target']
            weight = float(link.get('weight', 1.0))
            G.add_edge(source, target, weight=weight)
    return G

# --- Core Pathfinding Algorithm ---

def bounded_dijkstra(graph, start_node, end_node=None, targets=None, max_expansions=20000, time_budget=1.5):
    """
    Performs a Dijkstra search with limits on node expansions and execution time.
    """
    if start_node not in graph:
        return None, float('inf'), "start_node_missing"

    search_start_time = time.monotonic()
    pq = [(0, start_node)]
    distances = {start_node: 0}
    parents = {start_node: None}
    expansions = 0

    target_set = set(targets) if targets else set()
    if end_node:
        target_set.add(end_node)

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

    return None, float('inf'), "unreachable_or_limit"


# --- Landmark Preprocessing Functions ---

def select_landmarks(graph, k):
    """Selects k landmarks using degree-biased random sampling."""
    if not graph or graph.number_of_nodes() == 0:
        return []
    nodes = list(graph.nodes())
    degrees = [d for n, d in graph.degree(nodes)]
    total_degree = sum(degrees)
    if total_degree == 0:
        return random.sample(nodes, min(k, len(nodes)))

    probabilities = [d / total_degree for d in degrees]
    chosen = set()
    while len(chosen) < k and len(chosen) < len(nodes):
        selected_nodes = random.choices(nodes, weights=probabilities, k=k - len(chosen))
        chosen.update(selected_nodes)
    return list(chosen)

def precompute_landmark_paths(graph, landmarks):
    """Precomputes shortest paths between a subset of landmark pairs."""
    print("INFO: Starting landmark path precomputation...", file=sys.stderr)
    precomputed_paths = {}
    k = len(landmarks)
    subset_size = math.ceil(math.sqrt(k))
    
    # DYNAMIC LIMIT: Scale search effort with graph size.
    expansion_limit = max(15000, int(math.sqrt(graph.number_of_nodes()) * 10))
    print(f"INFO: Using expansion limit of {expansion_limit} for precomputation.", file=sys.stderr)

    for i, lm1 in enumerate(landmarks):
        for j in range(1, subset_size + 1):
            lm2 = landmarks[(i + j) % k]
            if lm1 == lm2 or (lm1, lm2) in precomputed_paths:
                continue
            
            path, cost, reason = bounded_dijkstra(graph, lm1, lm2, max_expansions=expansion_limit, time_budget=0.5)
            if reason == 'found':
                precomputed_paths[(lm1, lm2)] = (path, cost)
                precomputed_paths[(lm2, lm1)] = (path[::-1], cost)
                
    print(f"INFO: Precomputation found {len(precomputed_paths)//2} inter-landmark paths.", file=sys.stderr)
    return precomputed_paths

# --- Query Processing Functions ---

def get_query_pairs(query_file):
    """Generator to yield (source, dest) pairs from the query file."""
    with open(query_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            try:
                yield (int(parts[0]), int(parts[1]))
            except ValueError:
                continue

def process_small_graph_queries(G, queries_path, output_path):
    """Process queries for small graphs using built-in bidirectional Dijkstra."""
    print("INFO: Processing queries on small graph using bidirectional Dijkstra.", file=sys.stderr)
    with (open(output_path, 'w') if output_path != "-" else sys.stdout) as out_f:
        for source, dest in get_query_pairs(queries_path):
            start_timer = time.monotonic()
            try:
                cost, path = nx.bidirectional_dijkstra(G, source, dest, weight='weight')
                path_str = ' -> '.join(map(str, path))
                query_info = f"Query {source} -> {dest}"
                cost_info = f"Cost: {cost:.4f}"
                timer_info = f"Time: {time.monotonic() - start_timer:.4f}s"
                path_info = f"Path: {path_str}"
                output_line = f"{query_info} | {cost_info} | {timer_info}\n| {path_info}\n"
            except nx.NetworkXNoPath:
                output_line = f"Query {source} -> {dest} | Cost: inf | Reason: No path exists.\n"
            except nx.NodeNotFound as e:
                output_line = f"Query {source} -> {dest} | Cost: inf | Reason: {e}\n"
            
            out_f.write(output_line + "\n")

def find_nearest_landmark(graph, node, landmarks):
    """Finds the closest landmark to a given node using a bounded search."""
    # DYNAMIC LIMIT: Give this search a generous budget as it's critical.
    expansion_limit = max(20000, int(math.sqrt(graph.number_of_nodes()) * 20))
    path, cost, reason = bounded_dijkstra(graph, start_node=node, targets=set(landmarks), max_expansions=expansion_limit)
    if path:
        return path, cost, path[-1]
    return None, float('inf'), None

def stitch_path(p1, p2, p3):
    """Stitches three path segments together, removing overlapping nodes."""
    if not p1 or not p2 or not p3: return None
    full_path = list(p1)
    if p2 and p2[0] == full_path[-1]:
        full_path.extend(p2[1:])
    else: return None
    
    if p3 and p3[0] == full_path[-1]:
        full_path.extend(p3[1:])
    else: return None
    return full_path

def get_path_cost(graph, path):
    """Calculates the total weight of a path."""
    return sum(float(graph[u][v].get('weight', 1.0)) for u, v in zip(path[:-1], path[1:]))

def query_path_landmarks(graph, landmarks, landmark_paths, source, dest):
    """Handles a single query using the landmark strategy."""
    path_to_slm, _, source_lm = find_nearest_landmark(graph, source, landmarks)
    path_to_dlm, _, dest_lm = find_nearest_landmark(graph, dest, landmarks)

    if not source_lm or not dest_lm:
        return {'cost': float('inf'), 'path': [], 'reason': "Could not reach a landmark."}

    inter_lm_path = None
    if source_lm == dest_lm:
        inter_lm_path = [source_lm]
    elif (source_lm, dest_lm) in landmark_paths:
        inter_lm_path, _ = landmark_paths[(source_lm, dest_lm)]
    else:
        # FALLBACK with DYNAMIC LIMIT
        expansion_limit = max(20000, int(math.sqrt(graph.number_of_nodes()) * 20))
        path, _, reason = bounded_dijkstra(graph, source_lm, dest_lm, max_expansions=expansion_limit)
        if reason == 'found':
            inter_lm_path = path

    if not inter_lm_path:
        return {'cost': float('inf'), 'path': [], 'reason': "No path found between landmarks."}
        
    full_path = stitch_path(path_to_slm, inter_lm_path, path_to_dlm[::-1])
    if full_path:
        return {'cost': get_path_cost(graph, full_path), 'path': full_path}
    else:
        return {'cost': float('inf'), 'path': [], 'reason': "Path stitching failed."}


def process_large_graph_queries(G, landmarks, landmark_paths, queries_path, output_path):
    """Process queries for large graphs using the landmark routing strategy."""
    print("INFO: Processing queries on large graph using landmark routing.", file=sys.stderr)
    with (open(output_path, 'w') if output_path != "-" else sys.stdout) as out_f:
        for source, dest in get_query_pairs(queries_path):
            if source not in G or dest not in G:
                output_line = f"Query {source} -> {dest} | Cost: inf | Reason: Node not in graph.\n"
                out_f.write(output_line)
                continue
            
            result = query_path_landmarks(G, landmarks, landmark_paths, source, dest)
            cost = result['cost']
            
            if cost != float('inf'):
                path_str = ' -> '.join(map(str, result['path']))
                output_line = f"Query {source} -> {dest} | Cost: {cost:.4f} | Path: {path_str}\n"
            else:
                reason = result.get('reason', 'N/A')
                output_line = f"Query {source} -> {dest} | Cost: inf | Reason: {reason}\n"
            
            out_f.write(output_line)

# --- Main Execution ---
def main():
    global IS_SMALL_GRAPH
    parser = argparse.ArgumentParser(description="Pathfinder with Landmark Routing")
    parser.add_argument("-g", "--graph", required=True, help="Path to the node-link JSON graph file")
    parser.add_argument("-q", "--queries", required=False, help="Path to the queries file")
    parser.add_argument("-o", "--output", default="-", help="Output file path ('-' for stdout)")
    args = parser.parse_args()

    try:
        signal.signal(signal.SIGALRM, deadline_handler)
        signal.alarm(int(GLOBAL_DEADLINE_SECONDS))
    except AttributeError:
        print("WARNING: signal.alarm not available on this OS.", file=sys.stderr)

    print(f"--- PREPROCESSING (Max {GLOBAL_DEADLINE_SECONDS}s) ---", file=sys.stderr)
    G, landmarks, landmark_paths = None, [], {}
    
    try:
        IS_SMALL_GRAPH = os.path.getsize(args.graph) <= SIZE_THRESHOLD_BYTES
        
        if IS_SMALL_GRAPH:
            print(f"INFO: Loading small graph ({os.path.getsize(args.graph) / (1024*1024):.2f} MB) into memory...", file=sys.stderr)
            with open(args.graph, 'r') as f:
                data = json.load(f)
            G = nx.node_link_graph(data)
        else:
            print(f"INFO: Graph file is large ({os.path.getsize(args.graph) / (1024*1024):.2f} MB).", file=sys.stderr)
            G = load_graph_streaming(args.graph)
        
        # Added memory usage reporting and check right after loading.
        rss_after_load = psutil.Process(os.getpid()).memory_info().rss
        print(f"INFO: Graph loaded. Current memory usage: {rss_after_load / (1<<20):.2f} MB", file=sys.stderr)
        limit()
        
        if not IS_SMALL_GRAPH:
            print(f"INFO: Large graph ({G.number_of_nodes()} nodes). Starting landmark selection.", file=sys.stderr)
            k = math.ceil(math.sqrt(G.number_of_nodes()))
            landmarks = select_landmarks(G, k)
            landmark_paths = precompute_landmark_paths(G, landmarks)
            limit()

    except (TimeoutInterrupt, Exception) as e:
        if isinstance(e, TimeoutInterrupt):
             print("INFO: Preprocessing stopped by deadline.", file=sys.stderr)
        else:
             print(f"ERROR during preprocessing: {e}", file=sys.stderr)
        if G is None:
            print("FATAL: No graph was loaded. Exiting.", file=sys.stderr)
            return
    finally:
        signal.alarm(0)
        print(f"--- Preprocessing complete in {time.monotonic() - PROGRAM_START:.2f}s ---", file=sys.stderr)

    if G is not None:
        print("\n--- QUERYING ---", file=sys.stderr)
        if not args.queries:
            print("Please provide a queries file with -q option.", file=sys.stderr)
            args.queries = input("Enter the path to the queries file: ").strip()

        if IS_SMALL_GRAPH:
            process_small_graph_queries(G, args.queries, args.output)
        else:
            process_large_graph_queries(G, landmarks, landmark_paths, args.queries, args.output)
    
    print("\n--- Done ---", file=sys.stderr)

if __name__ == "__main__":
    main()

