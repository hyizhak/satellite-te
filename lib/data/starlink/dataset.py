from torch.utils.data import Dataset
import numpy as np

from ...asset import AssetManager
from .adapter import StarlinkAdapter

class StarlinkTopoTMDataset(Dataset):
    
    def __init__(self, problem_path, topo_idx, mode):
        self.problem_path = problem_path
        self.topo_idx = topo_idx

        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        if mode == 'train':
            self.size = AssetManager.tm_train_separate_num(problem_path, topo_idx)
        else:
            self.size = AssetManager.tm_test_separate_num(problem_path, topo_idx)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            file_path = AssetManager.tm_train_separate_path(self.problem_path, self.topo_idx, idx)
        else:
            file_path = AssetManager.tm_test_separate_path(self.problem_path, self.topo_idx, idx)
        return StarlinkAdapter.matrix_from_tm_file(file_path)
        