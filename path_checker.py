"""Path viability checker for PathFinder graphs (in-memory version).

This script fully loads a node-link JSON graph into NetworkX and verifies:

* Each query pair from a text file has a traversable path, and/or
* Each explicit path listed in a solver report is a valid sequence of edges
  whose weights sum to the advertised cost.

The checker avoids streaming or incremental growth; it assumes the graph fits
comfortably in memory and uses NetworkX Dijkstra searches together with
edge-by-edge validation of reported paths.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import networkx as nx


@dataclass
class QueryResult:
    """Stores the outcome for a single query."""

    source: int
    dest: int
    status: str
    elapsed: float
    cost: float = float("inf")
    path: Optional[Sequence[int]] = None
    reason: Optional[str] = None

    def format(self) -> str:
        """Pretty-print the result."""
        if self.status in {"path_found", "path_valid"} and self.path:
            path_str = " -> ".join(map(str, self.path))
            return (
                f"Query {self.source} -> {self.dest} | status=VALID | cost={self.cost:.4f} "
                f"| time={self.elapsed:.4f}s\n| Path: {path_str}"
            )
        reason = self.reason or "No path"
        return (
            f"Query {self.source} -> {self.dest} | status=MISSING | time={self.elapsed:.4f}s "
            f"| reason={reason}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Check whether queries have viable paths")
    parser.add_argument("-g", "--graph", required=True, help="Path to the node-link JSON graph")
    parser.add_argument("-q", "--queries", help="Path to the queries text file")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Where to write results ('-' for stdout)",
    )
    parser.add_argument(
        "-p",
        "--paths",
        help="Path to a solver output report to validate",
    )
    return parser.parse_args(argv)


def load_graph(graph_path: Path) -> nx.Graph:
    """Load the entire graph from a node-link JSON file."""

    with graph_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return nx.node_link_graph(data)


def iter_query_pairs(query_path: Path) -> Iterable[Tuple[int, int]]:
    """Yield (source, dest) pairs from the query file."""

    with query_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                yield int(parts[0]), int(parts[1])
            except ValueError:
                continue


def check_query(graph: nx.Graph, source: int, dest: int) -> QueryResult:
    """Run a weighted Dijkstra search to determine path viability."""

    start = time.monotonic()

    if source not in graph:
        return QueryResult(
            source=source,
            dest=dest,
            status="no_path",
            elapsed=time.monotonic() - start,
            reason=f"Source node {source} not present",
        )
    if dest not in graph:
        return QueryResult(
            source=source,
            dest=dest,
            status="no_path",
            elapsed=time.monotonic() - start,
            reason=f"Destination node {dest} not present",
        )

    try:
        path = nx.dijkstra_path(graph, source, dest, weight="weight")
        cost = nx.dijkstra_path_length(graph, source, dest, weight="weight")
        elapsed = time.monotonic() - start
        return QueryResult(
            source=source,
            dest=dest,
            status="path_found",
            elapsed=elapsed,
            cost=cost,
            path=path,
        )
    except nx.NetworkXNoPath:
        elapsed = time.monotonic() - start
        return QueryResult(
            source=source,
            dest=dest,
            status="no_path",
            elapsed=elapsed,
            reason="No path exists",
        )


ParsedPath = Tuple[int, int, float, Optional[List[int]]]


def parse_report_paths(report_path: Path) -> List[ParsedPath]:
    """Extract (source, dest, cost, path) tuples from a solver report."""

    query_re = re.compile(r"^Query\s+(\d+)\s*->\s*(\d+)\s*\|\s*Cost:\s*([0-9.]+|inf)")
    entries: List[ParsedPath] = []
    pending: Optional[Tuple[int, int, float]] = None

    with report_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            match = query_re.match(line)
            if match:
                src = int(match.group(1))
                dst = int(match.group(2))
                cost_str = match.group(3)
                cost = float("inf") if cost_str == "inf" else float(cost_str)
                pending = (src, dst, cost)
                # By default assume no explicit path unless we see a Path line next.
                entries.append((src, dst, cost, None))
                continue

            if line.startswith("| Path:") and entries:
                path_str = line.split(":", 1)[1].strip()
                if path_str:
                    nodes = [int(token.strip()) for token in path_str.split("->")]
                else:
                    nodes = []
                src, dst, cost, _ = entries[-1]
                entries[-1] = (src, dst, cost, nodes)

    return entries


def validate_report_path(graph: nx.Graph, source: int, dest: int, cost: float, path: Optional[Sequence[int]]) -> QueryResult:
    """Verify that the explicit path is consistent with the graph."""

    start_time = time.monotonic()
    if path is None:
        return QueryResult(
            source=source,
            dest=dest,
            status="path_invalid",
            elapsed=time.monotonic() - start_time,
            reason="No explicit path provided",
        )

    if not path:
        return QueryResult(
            source=source,
            dest=dest,
            status="path_invalid",
            elapsed=time.monotonic() - start_time,
            reason="Empty path sequence",
        )

    if path[0] != source or path[-1] != dest:
        return QueryResult(
            source=source,
            dest=dest,
            status="path_invalid",
            elapsed=time.monotonic() - start_time,
            path=path,
            reason="Path endpoints do not match query",
        )

    total_cost = 0.0
    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]
        if u not in graph or v not in graph:
            return QueryResult(
                source=source,
                dest=dest,
                status="path_invalid",
                elapsed=time.monotonic() - start_time,
                path=path,
                reason=f"Node {u if u not in graph else v} missing from graph",
            )
        if not graph.has_edge(u, v):
            return QueryResult(
                source=source,
                dest=dest,
                status="path_invalid",
                elapsed=time.monotonic() - start_time,
                path=path,
                reason=f"Edge {u}->{v} not present",
            )
        total_cost += float(graph[u][v].get("weight", 1.0))

    elapsed = time.monotonic() - start_time
    if cost != float("inf") and abs(total_cost - cost) > 1e-6:
        return QueryResult(
            source=source,
            dest=dest,
            status="path_invalid",
            elapsed=elapsed,
            path=path,
            cost=total_cost,
            reason=f"Reported cost {cost} differs from actual {total_cost:.4f}",
        )

    return QueryResult(
        source=source,
        dest=dest,
        status="path_valid",
        elapsed=elapsed,
        cost=total_cost,
        path=path,
    )


def summarize(results: Sequence[QueryResult]) -> str:
    """Provide a concise summary of overall results."""

    total = len(results)
    found = sum(1 for r in results if r.status in {"path_found", "path_valid"})
    missing = total - found
    avg_time = sum(r.elapsed for r in results) / total if total else 0.0
    return (
        f"Summary: total={total} | found={found} | missing={missing} | avg_time={avg_time:.4f}s"
    )


def write_results(results: Sequence[QueryResult], destination: str) -> None:
    """Write formatted results to stdout or a file."""

    lines = [r.format() for r in results]
    payload = "\n".join(lines + ["", summarize(results)])

    if destination == "-":
        print(payload)
    else:
        with open(destination, "w", encoding="utf-8") as fh:
            fh.write(payload)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the checker CLI."""

    args = parse_args(argv)
    graph_path = Path(args.graph)
    graph = load_graph(graph_path)
    results: List[QueryResult] = []

    if args.paths:
        report_entries = parse_report_paths(Path(args.paths))
        if not report_entries:
            print("WARNING: No query entries found in paths report.", file=sys.stderr)
        for src, dst, cost, path in report_entries:
            results.append(validate_report_path(graph, src, dst, cost, path))

    if args.queries:
        for source, dest in iter_query_pairs(Path(args.queries)):
            results.append(check_query(graph, source, dest))

    if not results:
        print("No queries or paths to process.", file=sys.stderr)
        return

    write_results(results, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
