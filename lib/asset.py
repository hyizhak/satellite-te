import os
import networkx as nx
import glob
import pickle

TL_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
TOPOLOGIES_DIR = os.path.join(TL_DIR, "topologies")
TM_DIR = os.path.join(TL_DIR, "traffic-matrices")

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

class AssetManager():

    TOPO_PREFIX = "topo_"
    
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
    def topo_num(cls, problem_path):
        return len(glob.glob(os.path.join(problem_path, f'{cls.TOPO_PREFIX}*')))

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
    def save_graph_(cls, problem_path, file_idx, tm_idx, G: nx.Graph):
        nx.write_gpickle(G, cls.graph_path(problem_path, tm_idx if file_idx == "A" else 5000 + tm_idx, True))

    @classmethod
    def tm_train_path(cls, problem_path, topo_idx, create_path=False):
        return os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), "tm_train.pkl")
    
    @classmethod
    def tm_train_separate_path(cls, problem_path, topo_idx, tm_idx, create_path=False):
        path = os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), 'tm_train', f"{tm_idx}.pkl")
        if create_path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    @classmethod
    def tm_train_separate_num(cls, problem_path, topo_idx):
        return len(glob.glob(os.path.join(cls._data_topo_dir(problem_path, topo_idx), 'tm_train', '*.pkl')))
    
    @classmethod
    def load_tm_train_separate(cls, problem_path, topo_idx, tm_idx):
        with open(cls.tm_train_separate_path(problem_path, topo_idx, tm_idx), 'rb') as f:
            return pickle.load(f)

    @classmethod
    def save_tm_train_separate_(cls, problem_path, topo_idx, tm_idx, tm):
        with open(cls.tm_train_separate_path(problem_path, topo_idx, tm_idx, True), 'wb') as f:
            pickle.dump(tm, f)

    @classmethod
    def tm_test_path(cls, problem_path, topo_idx, create_path=False):
        return os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), "tm_test.pkl")
    
    @classmethod
    def tm_test_separate_path(cls, problem_path, topo_idx, tm_idx, create_path=False):
        path = os.path.join(cls._data_topo_dir(problem_path, topo_idx, create_path), 'tm_test', f"{tm_idx}.pkl")
        if create_path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    @classmethod
    def tm_test_separate_num(cls, problem_path, topo_idx):
        return len(glob.glob(os.path.join(cls._data_topo_dir(problem_path, topo_idx), 'tm_test', '*.pkl')))
    
    @classmethod
    def load_tm_test_separate(cls, problem_path, topo_idx, tm_idx):
        with open(cls.tm_test_separate_path(problem_path, topo_idx, tm_idx), 'rb') as f:
            return pickle.load(f)
        
    @classmethod
    def save_tm_test_separate_(cls, problem_path, topo_idx, tm_idx, tm):
        with open(cls.tm_test_separate_path(problem_path, topo_idx, tm_idx, True), 'wb') as f:
            pickle.dump(tm, f)  

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
    def _pathform_metadata_path(cls, problem_path, topo_idx, create_path=True):
        return os.path.join(
            cls._data_topo_dir(problem_path, topo_idx, create_path),
            "path_metadata.pkl"
        )
        
    @classmethod
    def save_pathform_metadata_(cls, problem_path, topo_idx, meta):
        with open(cls._pathform_metadata_path(problem_path, topo_idx, True), 'wb') as f:
            pickle.dump(meta, f)

    @classmethod
    def save_pathform_metadata_(cls, problem_path, file_idx, tm_idx, meta):
        with open(cls._pathform_metadata_path(problem_path, tm_idx if file_idx == "A" else 5000 + tm_idx, True), 'wb') as f:
            pickle.dump(meta, f)
            
    @classmethod
    def load_pathform_metadata(cls, problem_path, topo_idx):
        with open(cls._pathform_metadata_path(problem_path, topo_idx, False), 'rb') as f:
            return pickle.load(f)
        
    @classmethod
    def save_starlink_dataset_(cls, output_path, file_path, starlink_dataset):
        with open(os.path.join(output_path, file_path), 'wb') as f:
            pickle.dump(starlink_dataset, f)
        
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
    
            