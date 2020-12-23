import os
import pickle
import numpy as np
from typing import Tuple

CITATION = 'data/transfer'
COAUTHOR = 'data/transfer3'


def encode_label(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    ret = np.zeros(shape=[n, 4], dtype=np.int)
    s2i = {
        'AI&DM': 0,
        'HC': 1,
        'CA': 2,
        'CN': 3
    }
    for i in range(n):
        ret[i, s2i[labels[i]]] = 1
    return ret


def norm_adj(adj: np.ndarray) -> np.ndarray:
    hat_a = adj + np.eye(adj.shape[0])
    hat_d = np.diag(np.power(np.sum(hat_a, axis=1), -1/2))
    hat_d[hat_d == np.inf] = 0
    return hat_d @ hat_a @ hat_d


def dump_citation(name: str):
    list_id_feature_label = np.genfromtxt(f'{CITATION}/{name}.content', dtype=np.str)
    features = np.array(list_id_feature_label[:, 1: -1], dtype=np.int)
    labels = list_id_feature_label[:, -1]
    print('labels:', set(labels))
    labels = encode_label(labels)

    n_node = list_id_feature_label.shape[0]
    edge_list = np.genfromtxt(f'{CITATION}/{name}.cites', dtype=np.int)
    adj = np.zeros(shape=[n_node, n_node], dtype=np.float)
    for u, v in edge_list:
        adj[v, u] = 1
    adj = norm_adj(adj)

    path = f'{CITATION}/{name}.pickle'
    with open(path, 'wb+') as fp:
        pickle.dump((adj, features, labels), fp)


def load_citation(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = f'{CITATION}/{name}.pickle'
    if not os.path.exists(path):
        dump_citation(name)
    with open(path, 'rb') as fp:
        adj, features, labels = pickle.load(fp)
    return adj, features, labels


def dump_coauthor(name: str):
    list_id_feature_label = np.genfromtxt(f'{COAUTHOR}/{name}.content', dtype=np.str)
    features = np.array(list_id_feature_label[:, 1: -1], dtype=np.int)
    labels = np.genfromtxt(f'{COAUTHOR}/{name}.multilabel', dtype=np.int)

    n_node = list_id_feature_label.shape[0]
    edge_list = np.genfromtxt(f'{COAUTHOR}/{name}.cites', dtype=np.int)
    adj = np.zeros(shape=[n_node, n_node], dtype=np.float)
    for u, v in edge_list:
        adj[v, u] = 1
    adj = norm_adj(adj)

    path = f'{COAUTHOR}/{name}.pickle'
    with open(path, 'wb+') as fp:
        pickle.dump((adj, features, labels), fp)


def load_coauthor(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = f'{COAUTHOR}/{name}.pickle'
    if not os.path.exists(path):
        dump_coauthor(name)
    with open(path, 'rb') as fp:
        adj, features, labels = pickle.load(fp)
    return adj, features, labels


if __name__ == '__main__':
    # dump_citation('chn')
    # dump_coauthor('usa')
    # dump_coauthor('chn')
    # dump_coauthor('usa')
    # a, f, l = load_citation('usa')
    # print(a.sum(axis=1))
    # print(f.sum(axis=1))
    # print(l.sum(axis=1))
    pass
