"""PathFinder command-line engine.

This tool loads a JSON node-link graph, picks a strategy based on its size,
and answers shortest-path queries. The high-level flow is:

1. Parse command-line arguments for the graph, query list, and output file.
2. Set a 60-second preprocessing budget and, when necessary, a 10-second
   per-query search budget.
3. Inspect the graph size to choose a loading strategy:
   * Small files load fully into memory. 
   * Large files stream directly into a NetworkX graph.
   * Huge files grow a sampled subgraph and keep expanding it until the
     deadline, always prioritising nodes from the query list.
4. Precompute lightweight landmark data for large-but-not-huge graphs.
5. Answer each query with the matching strategy:
   * Small graphs run bidirectional Dijkstra on the full graph.
   * Large graphs route via landmarks and stitched paths.
   * Huge graphs keep enlarging the streamed subgraph while searching and
     retry until a path is found or time expires.

Every stage enforces a memory cap, emits progress on stderr, and reports
clearly when a path cannot be found in the available time.
"""

import argparse
import time
import math
import random
import heapq
import sys
import os
import json
from collections import deque
import networkx as nx
import signal
import psutil
import ijson

# --- Global Configuration & Timers ---
PROGRAM_START = time.monotonic()
GLOBAL_DEADLINE_SECONDS = 60.0
SIZE_THRESHOLD_BYTES = 100 * 1024 * 1024  # 100 MB
HUGE_GRAPH_THRESHOLD_BYTES = 250 * 1024 * 1024  # 250 MB
EXTREME_GRAPH_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB
IS_SMALL_GRAPH = None
LIMIT = 1 << 30  # 1 GiB
MAX_QUERY_SEARCH_SECONDS = 10.0


# --- Custom Exception for Timeout ---
class TimeoutInterrupt(Exception):
    """Custom exception for deadline interrupts."""
    pass


def limit():
    """Enforce 1 GiB process memory cap."""
    rss = psutil.Process(os.getpid()).memory_info().rss
    if rss > LIMIT:
        sys.stderr.write(
            f"\nERROR: Memory limit exceeded ({rss / (1<<30):.2f} GiB > {LIMIT / (1<<30):.2f} GiB).\n"
        )
        sys.stderr.write("Consider reducing the working set or using a smaller subgraph.\n")
        sys.exit(137)


def deadline_handler(signum, frame):
    """Signal handler for global timeout."""
    print("INTERRUPT: Global deadline reached. Stopping preprocessing.", file=sys.stderr)
    raise TimeoutInterrupt


# --- Graph Loading & Streaming Utilities ---
def load_graph_streaming(path):
    """Load a graph using a streaming parser to keep memory usage in check."""
    print("INFO: Using streaming loader for large graph...", file=sys.stderr)
    graph = nx.Graph()
    with open(path, "rb") as fh:
        for link in ijson.items(fh, "links.item"):
            source = link.get("source")
            target = link.get("target")
            if source is None or target is None:
                continue
            weight = float(link.get("weight", 1.0))
            graph.add_edge(source, target, weight=weight)
    return graph


def sample_graph_nodes(path, max_samples=2048):
    """Collect a sample of node ids without materialising the whole graph."""
    samples = []
    seen = set()
    try:
        with open(path, "rb") as fh:
            for node in ijson.items(fh, "nodes.item"):
                node_id = node.get("id")
                if node_id is None or node_id in seen:
                    continue
                samples.append(node_id)
                seen.add(node_id)
                if len(samples) >= max_samples:
                    return samples
    except Exception:
        pass

    try:
        with open(path, "rb") as fh:
            for link in ijson.items(fh, "links.item"):
                for key in ("source", "target"):
                    node_id = link.get(key)
                    if node_id is None or node_id in seen:
                        continue
                    samples.append(node_id)
                    seen.add(node_id)
                    if len(samples) >= max_samples:
                        return samples
    except Exception:
        pass
    return samples


def get_max_node_id(path):
    """Fallback helper if no sampled nodes were found."""
    max_id = -1
    with open(path, "rb") as fh:
        for prefix, event, value in ijson.parse(fh):
            if prefix == "nodes.item.id":
                try:
                    candidate = int(value)
                except (ValueError, TypeError):
                    continue
                if candidate > max_id:
                    max_id = candidate
    if max_id >= 0:
        print(f"INFO: Max node id detected at {max_id}", file=sys.stderr)
    return max_id


def expand_subgraph_via_stream(path, graph, frontier, target_node_budget, max_passes=5):
    """Expand a subgraph by repeatedly scanning the JSON stream."""
    for node in frontier:
        graph.add_node(node)
    included = set(graph.nodes())
    newly_discovered = set()
    passes = 0

    while frontier and passes < max_passes:
        if len(included) >= target_node_budget:
            break
        passes += 1
        print(
            f"  -> Streaming pass {passes} over links (frontier size {len(frontier)})",
            file=sys.stderr,
        )
        new_frontier = set()
        allow_growth = len(included) < target_node_budget

        with open(path, "rb") as fh:
            for link in ijson.items(fh, "links.item"):
                source = link.get("source")
                target = link.get("target")
                if source is None or target is None:
                    continue

                source_included = source in included
                target_included = target in included
                source_frontier = source in frontier
                target_frontier = target in frontier

                if not (source_included or target_included or source_frontier or target_frontier):
                    continue

                weight = float(link.get("weight", 1.0))
                graph.add_edge(source, target, weight=weight)

                if not allow_growth:
                    continue

                if source_frontier and target not in included and len(included) < target_node_budget:
                    included.add(target)
                    new_frontier.add(target)
                    newly_discovered.add(target)
                if target_frontier and source not in included and len(included) < target_node_budget:
                    included.add(source)
                    new_frontier.add(source)
                    newly_discovered.add(source)

                if len(included) >= target_node_budget:
                    allow_growth = False

        frontier = new_frontier
        if not frontier:
            print("  -> Frontier exhausted during streaming expansion.", file=sys.stderr)

    return newly_discovered


def fill_subgraph_with_first_edges(path, graph, target_node_budget):
    """Fallback pass that greedily pulls edges until the budget is met."""
    included = set(graph.nodes())
    allow_new_component = not included

    if len(included) >= target_node_budget:
        return

    with open(path, "rb") as fh:
        for link in ijson.items(fh, "links.item"):
            source = link.get("source")
            target = link.get("target")
            if source is None or target is None:
                continue

            if allow_new_component or source in included or target in included:
                weight = float(link.get("weight", 1.0))
                graph.add_edge(source, target, weight=weight)
                included.add(source)
                included.add(target)
                allow_new_component = False

                if len(included) >= target_node_budget:
                    break


def build_unified_subgraph_on_disk(path, target_node_budget, seed_nodes=None):
    """Build a streaming-backed subgraph, prioritising the provided seeds."""
    graph = nx.Graph()
    seen_seeds = set()
    frontier_queue = deque()

    for node in seed_nodes or []:
        if node in seen_seeds:
            continue
        frontier_queue.append(node)
        seen_seeds.add(node)

    if not frontier_queue:
        samples = sample_graph_nodes(path, max_samples=max(target_node_budget * 2, 512))
        if not samples:
            max_id = get_max_node_id(path)
            if max_id == -1:
                print("ERROR: Could not discover any nodes in the graph file.", file=sys.stderr)
                return graph
            samples = list(range(max_id + 1))

        random.shuffle(samples)
        for node in samples:
            if node in seen_seeds:
                continue
            frontier_queue.append(node)
            seen_seeds.add(node)

    while frontier_queue and graph.number_of_nodes() < target_node_budget:
        batch = set()
        while frontier_queue and len(batch) < 32:
            batch.add(frontier_queue.popleft())

        if not batch:
            break

        print(
            f"INFO: Expanding streaming subgraph from seed batch (size {len(batch)}).",
            file=sys.stderr,
        )
        newly_found = expand_subgraph_via_stream(
            path,
            graph,
            frontier=batch,
            target_node_budget=target_node_budget,
        )
        limit()

        for node in newly_found:
            if node not in seen_seeds:
                frontier_queue.append(node)
                seen_seeds.add(node)

    if graph.number_of_nodes() < target_node_budget:
        print(
            f"INFO: Streaming builder reached {graph.number_of_nodes()} nodes (target {target_node_budget}).",
            file=sys.stderr,
        )
        fill_subgraph_with_first_edges(path, graph, target_node_budget)
        limit()
        print(
            f"INFO: Fallback expansion now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.",
            file=sys.stderr,
        )
    else:
        print(
            f"INFO: Streaming builder produced {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.",
            file=sys.stderr,
        )

    return graph


def build_subgraph_for_node(path, node_id):
    """Legacy helper that extracts all edges touching a node."""
    subgraph = nx.Graph()
    with open(path, "rb") as fh:
        for link in ijson.items(fh, "links.item"):
            source = link.get("source")
            target = link.get("target")
            if source == node_id or target == node_id:
                weight = float(link.get("weight", 1.0))
                subgraph.add_edge(source, target, weight=weight)
    return subgraph


# --- Core Pathfinding ---
def bounded_dijkstra(graph, start_node, end_node=None, targets=None, max_expansions=20000, time_budget=1.5):
    """Dijkstra with guards on expansions and wall-clock time."""
    if start_node not in graph:
        return None, float("inf"), "start_node_missing"

    search_start_time = time.monotonic()
    pq = [(0, start_node)]
    distances = {start_node: 0}
    parents = {start_node: None}
    expansions = 0

    target_set = set(targets) if targets else set()
    if end_node is not None:
        target_set.add(end_node)

    while pq and expansions < max_expansions:
        if time.monotonic() - search_start_time > time_budget:
            return None, float("inf"), "timeout"

        cost, current_node = heapq.heappop(pq)
        expansions += 1

        if cost > distances.get(current_node, float("inf")):
            continue

        if current_node in target_set:
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            return path[::-1], cost, "found"

        for neighbor, edge_data in graph[current_node].items():
            weight = float(edge_data.get("weight", 1.0))
            new_cost = cost + weight
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                parents[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))

    return None, float("inf"), "unreachable_or_limit"


def select_landmarks(graph, k):
    """Degree-weighted random landmark selection."""
    if graph.number_of_nodes() == 0:
        return []
    nodes = list(graph.nodes())
    degrees = [graph.degree(n) for n in nodes]
    total_degree = sum(degrees)
    if total_degree == 0:
        return random.sample(nodes, min(k, len(nodes)))

    probabilities = [deg / total_degree for deg in degrees]
    chosen = set()
    while len(chosen) < min(k, len(nodes)):
        selected = random.choices(nodes, weights=probabilities, k=k - len(chosen))
        chosen.update(selected)
    return list(chosen)


def precompute_landmark_paths(graph, landmarks):
    """Precompute a sparse set of inter-landmark shortest paths."""
    print("INFO: Starting landmark path precomputation...", file=sys.stderr)
    precomputed_paths = {}
    k = len(landmarks)
    if k == 0:
        return precomputed_paths

    subset_size = math.ceil(math.sqrt(k))
    expansion_limit = max(15000, int(math.sqrt(graph.number_of_nodes()) * 10))
    print(f"INFO: Using expansion limit {expansion_limit} for landmark searches.", file=sys.stderr)

    for i, lm1 in enumerate(landmarks):
        for j in range(1, subset_size + 1):
            lm2 = landmarks[(i + j) % k]
            if lm1 == lm2 or (lm1, lm2) in precomputed_paths:
                continue

            path, cost, reason = bounded_dijkstra(
                graph,
                lm1,
                lm2,
                max_expansions=expansion_limit,
                time_budget=0.5,
            )
            if reason == "found":
                precomputed_paths[(lm1, lm2)] = (path, cost)
                precomputed_paths[(lm2, lm1)] = (path[::-1], cost)

    print(
        f"INFO: Precomputation produced {len(precomputed_paths) // 2} landmark pairs.",
        file=sys.stderr,
    )
    return precomputed_paths


def find_nearest_landmark(graph, node, landmarks):
    """Locate the closest landmark from a node using a bounded search."""
    expansion_limit = max(20000, int(math.sqrt(graph.number_of_nodes()) * 20))
    path, cost, reason = bounded_dijkstra(
        graph,
        start_node=node,
        targets=set(landmarks),
        max_expansions=expansion_limit,
    )
    if reason == "found" and path:
        return path, cost, path[-1]
    return None, float("inf"), None


def stitch_path(p1, p2, p3):
    """Combine three path segments, removing duplicate joints."""
    if not p1 or not p2 or not p3:
        return None
    stitched = list(p1)
    if p2[0] == stitched[-1]:
        stitched.extend(p2[1:])
    else:
        return None
    if p3[0] == stitched[-1]:
        stitched.extend(p3[1:])
    else:
        return None
    return stitched


def get_path_cost(graph, path):
    """Compute total path cost from edge weights."""
    return sum(float(graph[u][v].get("weight", 1.0)) for u, v in zip(path[:-1], path[1:]))


def query_path_landmarks(graph, landmarks, landmark_paths, source, dest):
    """Answer a query via the landmark strategy."""
    path_to_slm, _, source_lm = find_nearest_landmark(graph, source, landmarks)
    path_to_dlm, _, dest_lm = find_nearest_landmark(graph, dest, landmarks)

    if not source_lm or not dest_lm:
        return {"cost": float("inf"), "path": [], "reason": "Could not reach a landmark."}

    if source_lm == dest_lm:
        inter_path = [source_lm]
    elif (source_lm, dest_lm) in landmark_paths:
        inter_path, _ = landmark_paths[(source_lm, dest_lm)]
    else:
        expansion_limit = max(20000, int(math.sqrt(graph.number_of_nodes()) * 20))
        inter_path, _, reason = bounded_dijkstra(
            graph,
            source_lm,
            dest_lm,
            max_expansions=expansion_limit,
        )
        if reason != "found":
            inter_path = None

    if not inter_path:
        return {"cost": float("inf"), "path": [], "reason": "No path found between landmarks."}

    full_path = stitch_path(path_to_slm, inter_path, path_to_dlm[::-1])
    if not full_path:
        return {"cost": float("inf"), "path": [], "reason": "Path stitching failed."}

    return {"cost": get_path_cost(graph, full_path), "path": full_path}


# --- Query Processing ---
def get_query_pairs(query_file):
    with open(query_file, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                yield int(parts[0]), int(parts[1])
            except ValueError:
                continue


def collect_query_nodes(query_file):
    """Return the set of node ids referenced by the query list."""
    nodes = set()
    try:
        for source, dest in get_query_pairs(query_file):
            nodes.add(source)
            nodes.add(dest)
    except OSError as exc:
        print(f"WARNING: Could not read queries for seeding ({exc}).", file=sys.stderr)
    return nodes


def process_small_graph_queries(graph, queries_path, output_path):
    print("INFO: Processing queries on small graph using bidirectional Dijkstra.", file=sys.stderr)
    with (open(output_path, "w") if output_path != "-" else sys.stdout) as out_f:
        for source, dest in get_query_pairs(queries_path):
            start_timer = time.monotonic()
            try:
                cost, path = nx.bidirectional_dijkstra(graph, source, dest, weight="weight")
                path_str = " -> ".join(map(str, path))
                elapsed = time.monotonic() - start_timer
                output_line = (
                    f"Query {source} -> {dest} | Cost: {cost:.4f} | Time: {elapsed:.4f}s\n| Path: {path_str}\n"
                )
            except nx.NetworkXNoPath:
                output_line = f"Query {source} -> {dest} | Cost: inf | Reason: No path exists.\n"
            except nx.NodeNotFound as exc:
                output_line = f"Query {source} -> {dest} | Cost: inf | Reason: {exc}\n"
            out_f.write(output_line + "\n")


def process_large_graph_queries(
    graph,
    landmarks,
    landmark_paths,
    queries_path,
    output_path,
    is_huge_graph,
    graph_path=None,
    target_node_budget=None,
):
    strategy = "unbounded search on unified subgraph" if is_huge_graph else "landmark routing"
    print(f"INFO: Processing queries on large graph using {strategy}.", file=sys.stderr)

    with (open(output_path, "w") if output_path != "-" else sys.stdout) as out_f:
        dynamic_budget = target_node_budget
        for source, dest in get_query_pairs(queries_path):
            missing_nodes = {node for node in (source, dest) if node not in graph}

            if is_huge_graph and missing_nodes and graph_path:
                desired_budget = dynamic_budget or graph.number_of_nodes() + 512
                desired_budget = max(desired_budget, graph.number_of_nodes() + len(missing_nodes) * 16)
                dynamic_budget = desired_budget

                print(
                    f"INFO: Expanding subgraph on-demand for nodes {sorted(missing_nodes)}.",
                    file=sys.stderr,
                )
                expand_subgraph_via_stream(
                    graph_path,
                    graph,
                    frontier=missing_nodes,
                    target_node_budget=dynamic_budget,
                )
                limit()
                missing_nodes = {node for node in (source, dest) if node not in graph}

            if missing_nodes:
                out_f.write(
                    f"Query {source} -> {dest} | Cost: inf | Reason: Node not in (partial) graph.\n"
                )
                continue

            if is_huge_graph:
                query_start = time.monotonic()
                search_deadline = query_start + MAX_QUERY_SEARCH_SECONDS
                attempt = 0
                path = None
                cost = float("inf")
                elapsed = 0.0
                failure_reason = "No path exists in the loaded subgraph."

                while time.monotonic() < search_deadline:
                    attempt += 1
                    try:
                        cost, path = nx.bidirectional_dijkstra(graph, source, dest, weight="weight")
                        elapsed = time.monotonic() - query_start
                        failure_reason = "found"
                        break
                    except nx.NetworkXNoPath:
                        failure_reason = "No path exists in the loaded subgraph."
                    except nx.NodeNotFound:
                        failure_reason = "Node disappeared from the current subgraph."

                    if time.monotonic() >= search_deadline:
                        break

                    if not graph_path:
                        break

                    expand_frontier = {source, dest}
                    desired_budget = dynamic_budget or graph.number_of_nodes() + 512
                    desired_budget = max(desired_budget, graph.number_of_nodes() + attempt * 256)
                    dynamic_budget = desired_budget
                    print(
                        f"INFO: Expanding subgraph during query {source}->{dest} attempt {attempt}.",
                        file=sys.stderr,
                    )
                    expand_subgraph_via_stream(
                        graph_path,
                        graph,
                        frontier=expand_frontier,
                        target_node_budget=dynamic_budget,
                    )
                    limit()

                total_elapsed = time.monotonic() - query_start

                if path:
                    path_str = " -> ".join(map(str, path))
                    output_line = (
                        f"Query {source} -> {dest} | Cost: {cost:.4f} | Time: {elapsed:.4f}s\n| Path: {path_str}\n"
                    )
                else:
                    output_line = (
                        f"Query {source} -> {dest} | Cost: inf | Reason: {failure_reason} (search timed out after {total_elapsed:.2f}s, {attempt} attempts).\n"
                    )
                out_f.write(output_line)
            else:
                result = query_path_landmarks(graph, landmarks, landmark_paths, source, dest)
                cost = result["cost"]
                if cost != float("inf"):
                    path_str = " -> ".join(map(str, result["path"]))
                    out_f.write(f"Query {source} -> {dest} | Cost: {cost:.4f} | Path: {path_str}\n")
                else:
                    reason = result.get("reason", "N/A")
                    out_f.write(f"Query {source} -> {dest} | Cost: inf | Reason: {reason}\n")


# --- Main Execution ---
def main():
    global IS_SMALL_GRAPH

    parser = argparse.ArgumentParser(description="Pathfinder with Landmark Routing")
    parser.add_argument("-g", "--graph", required=True, help="Path to the node-link JSON graph file")
    parser.add_argument("-q", "--queries", help="Path to the queries file")
    parser.add_argument("-o", "--output", default="-", help="Output file path ('-' for stdout)")
    args = parser.parse_args()

    try:
        signal.signal(signal.SIGALRM, deadline_handler)
        signal.alarm(int(GLOBAL_DEADLINE_SECONDS))
    except AttributeError:
        print("WARNING: signal.alarm not available on this OS.", file=sys.stderr)

    print(f"--- PREPROCESSING (Max {GLOBAL_DEADLINE_SECONDS}s) ---", file=sys.stderr)

    graph = None
    landmarks = []
    landmark_paths = {}
    is_huge_graph = False
    huge_graph_budget = None
    query_nodes = set()

    try:
        preprocess_start = time.monotonic()
        file_size = os.path.getsize(args.graph)
        is_huge_graph = file_size > HUGE_GRAPH_THRESHOLD_BYTES
        IS_SMALL_GRAPH = file_size <= SIZE_THRESHOLD_BYTES

        if args.queries:
            query_nodes = collect_query_nodes(args.queries)

        if is_huge_graph:
            print(
                f"INFO: Graph file is huge ({file_size / (1024*1024):.2f} MB). Using streaming subgraph builder.",
                file=sys.stderr,
            )
            huge_graph_budget = max(1000, min(20000, max(500, len(query_nodes) * 8))) if query_nodes else 1000
            graph = build_unified_subgraph_on_disk(
                args.graph,
                target_node_budget=huge_graph_budget,
                seed_nodes=query_nodes,
            )
            IS_SMALL_GRAPH = False
            if file_size > EXTREME_GRAPH_THRESHOLD_BYTES:
                print(
                    "INFO: Extreme graph size detected; continuing subgraph expansion until deadline.",
                    file=sys.stderr,
                )
                last_growth = graph.number_of_nodes()
                while True:
                    time_left = GLOBAL_DEADLINE_SECONDS - (time.monotonic() - preprocess_start)
                    if time_left <= 1.0:
                        break
                    if not graph.number_of_nodes():
                        break
                    seed_batch = set(random.sample(list(graph.nodes()), min(32, graph.number_of_nodes())))
                    target_size = graph.number_of_nodes() + max(128, graph.number_of_nodes() // 10)
                    expand_subgraph_via_stream(
                        args.graph,
                        graph,
                        frontier=seed_batch,
                        target_node_budget=target_size,
                    )
                    limit()
                    current_nodes = graph.number_of_nodes()
                    if current_nodes == last_growth:
                        # Attempt to inject new seeds from disk if no growth occurred.
                        additional_seeds = sample_graph_nodes(args.graph, max_samples=256)
                        random.shuffle(additional_seeds)
                        for node in additional_seeds[:64]:
                            seed_batch.add(node)
                        expand_subgraph_via_stream(
                            args.graph,
                            graph,
                            frontier=seed_batch,
                            target_node_budget=current_nodes + 256,
                        )
                        limit()
                        current_nodes = graph.number_of_nodes()
                        if current_nodes == last_growth:
                            break
                    last_growth = current_nodes
        elif IS_SMALL_GRAPH:
            print(
                f"INFO: Loading small graph ({file_size / (1024*1024):.2f} MB) into memory...",
                file=sys.stderr,
            )
            with open(args.graph, "r") as fh:
                data = json.load(fh)
            graph = nx.node_link_graph(data)
        else:
            print(
                f"INFO: Graph file is large ({file_size / (1024*1024):.2f} MB). Using streaming loader.",
                file=sys.stderr,
            )
            graph = load_graph_streaming(args.graph)

        limit()

        if graph and not IS_SMALL_GRAPH and not is_huge_graph:
            print(
                f"INFO: Large graph ({graph.number_of_nodes()} nodes). Starting landmark selection.",
                file=sys.stderr,
            )
            k = math.ceil(math.sqrt(graph.number_of_nodes())) if graph.number_of_nodes() > 0 else 0
            landmarks = select_landmarks(graph, k)
            landmark_paths = precompute_landmark_paths(graph, landmarks)
            limit()

    except (TimeoutInterrupt, Exception) as exc:
        if isinstance(exc, TimeoutInterrupt):
            print("INFO: Preprocessing stopped by deadline.", file=sys.stderr)
        else:
            print(f"ERROR during preprocessing: {exc}", file=sys.stderr)
        if graph is None:
            print("FATAL: No graph was loaded. Exiting.", file=sys.stderr)
            return
    finally:
        signal.alarm(0)
        print(f"--- Preprocessing complete in {time.monotonic() - PROGRAM_START:.2f}s ---", file=sys.stderr)

    if graph is not None:
        print("\n--- QUERYING ---", file=sys.stderr)
        if not args.queries:
            args.queries = input("Enter the path to the queries file: ").strip()

        if IS_SMALL_GRAPH:
            process_small_graph_queries(graph, args.queries, args.output)
        else:
            if is_huge_graph and args.queries:
                query_nodes = collect_query_nodes(args.queries)
                missing_seed_nodes = {node for node in query_nodes if node not in graph}
                if missing_seed_nodes:
                    desired_budget = max(
                        huge_graph_budget or 0,
                        graph.number_of_nodes() + len(missing_seed_nodes) * 16,
                        max(1000, min(20000, max(500, len(query_nodes) * 8))),
                    )
                    huge_graph_budget = desired_budget
                    print(
                        f"INFO: Expanding subgraph to cover query seeds ({len(missing_seed_nodes)} missing).",
                        file=sys.stderr,
                    )
                    expand_subgraph_via_stream(
                        args.graph,
                        graph,
                        frontier=missing_seed_nodes,
                        target_node_budget=huge_graph_budget,
                    )
                    limit()

            process_large_graph_queries(
                graph,
                landmarks,
                landmark_paths,
                args.queries,
                args.output,
                is_huge_graph,
                graph_path=args.graph if is_huge_graph else None,
                target_node_budget=huge_graph_budget,
            )

    print("\n--- Done ---", file=sys.stderr)


if __name__ == "__main__":
    main()
