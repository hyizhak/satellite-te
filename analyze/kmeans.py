import pickle
import torch
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
import numpy as np
import argparse

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

    test = list(range(len(tensor_list)))

    print("Representative tensor indices for each cluster:", [test[i] for i in representative_indices])

    return representative_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity", type=int, default=25)
    parser.add_argument("--mode", type=str, default="ISL")
    parser.add_argument("--num_clusters", type=int, default=8)
    args = parser.parse_args()
    intensity = args.intensity
    mode = args.mode
    num_clusters = args.num_clusters
    fpath = f'/data/projects/11003765/sate/satte/satellite-te/output/isomorphism_pruning/DataSetForSaTE{intensity}_{mode}_spaceTE/models/topoloy-GNN-embeddings/topologies.pkl'
    kmeans_embedding(fpath, num_clusters)