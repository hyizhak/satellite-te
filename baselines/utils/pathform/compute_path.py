from itertools import tee, islice
from sys import maxsize
import networkx as nx
import os
import argparse
import multiprocessing as mp
import pickle
import sys
import glob

ROOT = f'{os.path.dirname(os.path.abspath(__file__))}/../..'
sys.path.append(ROOT)

from lib import AssetManager
    
def compute_path(G: nx.Graph, num_path: int, edge_disjoint: bool, dist_metric: str):
    """Return path dictionary through computation."""
    path_dict = {}
    G = _graph_copy_with_edge_weights(G, dist_metric)
    for s_k in G.nodes:
        for t_k in G.nodes:
            if s_k == t_k:
                continue
            paths = _find_paths(G, s_k, t_k, num_path, edge_disjoint)
            paths_no_cycles = [remove_cycles(path) for path in paths]
            path_dict[(s_k, t_k)] = paths_no_cycles
    return path_dict

def _graph_copy_with_edge_weights(_G, dist_metric):
    G = _G.copy()

    if dist_metric == "inv-cap":
        for u, v, cap in G.edges.data("capacity"):
            if cap < 0.0:
                cap = 0.0
            try:
                G[u][v]["weight"] = 1.0 / cap
            except ZeroDivisionError:
                G[u][v]["weight"] = maxsize
    elif dist_metric == "min-hop":
        for u, v, cap in G.edges.data("capacity"):
            if cap <= 0.0:
                G[u][v]["weight"] = maxsize
            else:
                G[u][v]["weight"] = 1.0
    else:
        raise Exception("invalid dist_metric: {}".format(dist_metric))

    return G

def _find_paths(G, s_k, t_k, num_paths, disjoint=True, include_weight=False):
    def compute_weight(G, path):
        return sum(G[u][v]["weight"] for u, v in path_to_edge_list(path))

    def k_shortest_paths(G, source, target, k, weight="weight"):
        try:
            # Yen's shortest path algorithm
            return list(
                islice(nx.shortest_simple_paths(
                    G, source, target, weight=weight), k)
            )
        except nx.NetworkXNoPath:
            return []

    def k_shortest_edge_disjoint_paths(G, source, target, k, weight="weight"):
        def compute_distance(path):
            return sum(G[u][v][weight] for u, v in path_to_edge_list(path))

        return [
            remove_cycles(path)
            for path in sorted(
                nx.edge_disjoint_paths(G, s_k, t_k),
                key=lambda path: compute_distance(path),
            )[:k]
        ]

    if disjoint:
        if include_weight:
            return [
                (path, compute_weight(path))
                for path in k_shortest_edge_disjoint_paths(
                    G, s_k, t_k, num_paths, weight="weight"
                )
            ]
        else:
            return k_shortest_edge_disjoint_paths(
                G, s_k, t_k, num_paths, weight="weight"
            )
    else:
        if include_weight:
            return [
                (path, compute_weight(path))
                for path in k_shortest_paths(
                    G, s_k, t_k, num_paths, weight="weight")
            ]
        else:
            return k_shortest_paths(G, s_k, t_k, num_paths, weight="weight")

def path_to_edge_list(path):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(path)
    next(b, None)
    return zip(a, b)

# Remove cycles from path
def remove_cycles(path):
    stack = []
    visited = set()
    for node in path:
        if node in visited:
            # remove elements from this cycle
            while stack[-1] != node:
                visited.remove(stack[-1])
                stack = stack[:-1]
        else:
            stack.append(node)
            visited.add(node)
    return stack

def _compute_process(problem_path, topo_idx, num_path, edge_disjoint, dist_metric):
    fname = AssetManager.pathform_path(problem_path, topo_idx, num_path, edge_disjoint, dist_metric, create_path=True)
    print(f'Computing path: {fname}')
    G = AssetManager.load_graph(problem_path, topo_idx)
    path_dict = compute_path(G, num_path, edge_disjoint, dist_metric)
    with open(fname, 'wb') as f:
        pickle.dump(path_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--num-path', type=int, default=5)
    parser.add_argument('--edge-disjoint', type=bool, default=False)
    parser.add_argument('--dist-metric', type=str, default='min-hop', choices=['min-hop', 'inv-cap'])
    parser.add_argument('--parallel', type=int, default=None)
    
    # get all the subdirectories inside the problem path
    args = parser.parse_args()
    
    problem_path = args.problem_path
    num_path = args.num_path
    edge_disjoint = args.edge_disjoint
    dist_metric = args.dist_metric
    parallel = args.parallel
    
    print('Computing path for problem: {}'.format(problem_path))

    topo_num = len(glob.glob(os.path.join(problem_path, 'topo*')))
    
    params = []
    for i in range(topo_num):
        params.append((problem_path, i, num_path, edge_disjoint, dist_metric))
    
    with mp.Pool(parallel) as pool:
        pool.starmap(_compute_process, params)
        
    print('Path computation done.')
    