import pathlib
import pickle

import networkx as nx
import pandas as pd
import torch
import numpy as np
from node2vec import Node2Vec

from .adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix


def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def dump_pkl(obj: object, file_path: str):
    """Dumplicate pickle data.

    Args:
        obj (object): object
        file_path (str): file path
    """

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_adj(file_path: str, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == "original":
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx


def load_node2vec_emb(file_path: str) -> torch.Tensor:
    """load node2vec embedding

    Args:
        file_path (str): file path

    Returns:
        torch.Tensor: node2vec embedding
    """

    # spatial embedding
    with open(file_path, mode="r") as f:
        lines = f.readlines()
        temp = lines[0].split(" ")
        num_vertex, dims = int(temp[0]), int(temp[1])
        spatial_embeddings = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(" ")
            index = int(temp[0])
            spatial_embeddings[index] = torch.Tensor([float(ch) for ch in temp[1:]])
    return spatial_embeddings.cuda()


def get_node2vec(dataset,dim):

    SE_dir = 'datasets/{0}/output/adj/SE_{0}_{1}.txt'.format(dataset, dim)
    path = pathlib.Path(SE_dir)
    if not path.exists():
        if dataset == "PEMSBAY" or dataset == "METRLA":
            adj_info = load_pkl("datasets/{0}/raw_data/adj_{0}.pkl".format(dataset))
            adj_mx = adj_info[2]
        elif dataset == "ShenZhen":
            adj_mx = pd.read_csv("datasets/{0}/raw_data/adj_{0}.csv".format(dataset), header=None).values
        else:
            adj_mx = load_pkl("datasets/{0}/raw_data/adj_{0}.pkl".format(dataset))
        graph = nx.from_numpy_matrix(adj_mx)
        node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=200, workers=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        model.wv.save_word2vec_format(SE_dir)

    SE = load_node2vec_emb(SE_dir)
    return SE