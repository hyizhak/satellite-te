import os
import networkx as nx
import json

TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
TOPOLOGIES_DIR = os.path.join(TL_DIR, "topologies")
TM_DIR = os.path.join(TL_DIR, "traffic-matrices")

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

class AssetManager():
    
    @classmethod
    def _data_dir(cls, problem_path):
        return problem_path
    
    @classmethod
    def _data_topo_dir(cls, problem_path, topo_idx, create=False):
        path = os.path.join(cls._data_dir(problem_path), f"topo_{topo_idx}")
        if create:
            os.makedirs(path, exist_ok=True)
        return path
    
    @classmethod
    def graph_path(cls, problem_path, topo_idx, create_path=False):
        return os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), "graph.gpickle")

    @classmethod
    def load_graph(cls, problem_path, topo_idx) -> nx.Graph:
        return nx.read_gpickle(cls.graph_path(problem_path, topo_idx))
    
    @classmethod
    def save_graph_(cls, problem_path, topo_idx, G: nx.Graph):
        nx.write_gpickle(G, cls.graph_path(problem_path, topo_idx, True))

    @classmethod
    def tm_train_path(cls, problem_path, topo_idx, create_path=False):
        return os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), "tm_train.pkl")

    @classmethod
    def tm_test_path(cls, problem_path, topo_idx, create_path=False):
        return os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), "tm_test.pkl")

    @classmethod
    def pathform_path(cls, problem_path, topo_idx, num_path, edge_disjoint, dist_metric, create_path=False):
        return os.path.join(
            cls._data_topo_dir(problem_path, topo_idx, create_path), 
            f"paths_num-{num_path}_edge-disjoint-{edge_disjoint}_dist-metric-{dist_metric}-dict.pkl"
        )
    
    @classmethod
    def pathform_path(cls, graph_path, num_path, edge_disjoint, dist_metric, create_path=False):
        return os.path.join(os.path.dirname(graph_path), f"paths_num-{num_path}_edge-disjoint-{edge_disjoint}_dist-metric-{dist_metric}-dict.pkl"
        )
        
    @classmethod
    def model_dir(cls, work_dir, topo_idx=None, create_dir=False):
        if topo_idx is None:
            path = os.path.join(work_dir, 'models')
        else:
            path = os.path.join(work_dir, 'models', f'{topo_idx}')
        if create_dir:
            os.makedirs(path, exist_ok=True)
        return path
            
    @classmethod
    def train_log_dir(cls, work_dir, create_dir=False):
        path = os.path.join(work_dir, 'train_logs')
        if create_dir:
            os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def test_log_dir(cls, work_dir, create_dir=False):
        path = os.path.join(work_dir, 'test_logs')
        if create_dir:
            os.makedirs(path, exist_ok=True)
        return path
    
            