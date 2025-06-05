import os
import sys
import pickle
from sklearn.cluster import KMeans
import torch
import numpy as np

_ROOT = f'{os.path.dirname(__file__)}/..'

# ========== Benchmarking arguments
# Input and output
ARG_PROBLEM_PATH = f'{_ROOT}/input/IridiumDataSet14day20sec_Int5'
SOLUTION_PATH = f'{_ROOT}/input/lp_solutions'
ARG_TOPO_NUM = 10
ARG_OUTPUT_DIR = f'{_ROOT}/output'
ARG_OUTPUT_PREFIX = None
# Path forming parameters
ARG_PATH_NUM = 5
ARG_EDGE_DISJOINT = False
ARG_DIST_METIRC = "min-hop"

ARG_OBJ = "teal_total_flow"
ARG_LOSS = "kl_div"
ARG_SCALE_FACTOR = 1.0

ARG_TEST_TM_PER_TOPO = None

# ==========

TEST_HEADERS = [
            "topo_idx",
            "tm_idx",
            "total_demand",
            "obj_val",
            "ratio",
            "pre_runtime",
            "model_runtime",
            "post_runtime"
        ]
TEST_PLACEHOLDER = ','.join(['{}' for _ in TEST_HEADERS])

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib import AssetManager

import logging
LOGGING_FORMAT = "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt="%d/%b/%Y %H:%M:%S",
)

OBJ_STRS = ["teal_total_flow", "rounded_total_flow", "total_flow", "teal_min_max_link_util"]
LOSS_STRS = ["kl_div", "wasserstein"]
SCALE_FACTORS = [1.0]

def update_output_path(args, model):
    if args.output_prefix is None:
        parent_basename = os.path.basename(os.path.dirname(args.problem_path))
        problem_basename = os.path.basename(args.problem_path)
        args.output_prefix = f'{parent_basename}_{problem_basename}_{model}'
    args.work_dir = os.path.join(args.output_dir, args.output_prefix)

def read_solutions(file_path, smoothing=0.1):
    solutions = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                # Load each solution sequentially
                sol = pickle.load(f)
                if smoothing > 0:
                    sol = sol * (1 - smoothing) + smoothing / sol.shape[-1]                    
                solutions.append(sol)
            except EOFError:
                # End of file reached
                break
    return solutions

def kmeans_embedding(fpath, num_clusters=5):

    # read the pickle file
    with open(fpath, 'rb') as f:
        tensor_list = pickle.load(f)

    print("Number of tensors:", len(tensor_list))
    print("Shape of each tensor:", tensor_list[0].shape)


    # 1. Convert the tensor list into a 2D matrix
    # Assume each tensor has the same shape, e.g., (10, 5)
    tensor_matrix = torch.stack(tensor_list)  # (100, 10, 5)

    # Flatten the tensors to a 2D matrix of shape (100, 50) to apply KMeans
    flattened_tensor_matrix = tensor_matrix.view(tensor_matrix.size(0), -1).numpy()

    # 2. Apply sklearn's KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(flattened_tensor_matrix)

    # 3. Get the cluster labels for each tensor and the cluster centers
    cluster_labels = kmeans.labels_  # The cluster label for each tensor
    cluster_centers = kmeans.cluster_centers_  # The centers of each cluster

    # 4. Find the representative tensor for each cluster (the one closest to the cluster center)
    representative_indices = []
    for i in range(kmeans.n_clusters):
        # Get the indices of all tensors that belong to the ith cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        
        # Compute the distance of each tensor to the cluster center
        distances = np.linalg.norm(flattened_tensor_matrix[cluster_indices] - cluster_centers[i], axis=1)
        
        # Find the index of the tensor closest to the cluster center
        representative_index = cluster_indices[np.argmin(distances)]
        representative_indices.append(representative_index)

    # 5. Output the indices of the representative tensors for each cluster
    print("Representative tensor indices for each cluster:", representative_indices)

    return representative_indices
