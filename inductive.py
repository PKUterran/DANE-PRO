import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import f1_score

from reader import load_citation, load_coauthor
from net.gcn import GCN, Classifier

LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 1e-1
WEIGHT_DECAY = 5e-4
N_EPOCH = 200
N_EVAL = 20
RANDOM_SEED = 0


def set_seed(seed):
    torch.manual_seed(seed)


def train_single(start: str, use_tqdm=False):
    print(f'\tStart from graph {start}:')
    if start == 'A':
        src, tgt = 'chn', 'usa'
    elif start == 'B':
        src, tgt = 'usa', 'chn'
    else:
        assert False
    adj_a, features_a, labels_a = load_citation(src)
    adj_b, features_b, labels_b = load_citation(tgt)
    n_f = features_a.shape[1]
    n_l = labels_a.shape[0]
    adj_a = torch.from_numpy(adj_a).type(torch.float32)
    adj_b = torch.from_numpy(adj_b).type(torch.float32)
    features_a = torch.from_numpy(features_a).type(torch.float32)
    features_b = torch.from_numpy(features_b).type(torch.float32)
    labels_a = torch.from_numpy(labels_a).type(torch.float32)
    labels_b = torch.from_numpy(labels_b).type(torch.float32)

    gcn = GCN(n_f, 64, 64)
    classifier = Classifier(64, n_l)
    optimizer = optim.Adam(
        params=chain(gcn.parameters(), classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, N_EVAL, gamma=1 - LEARNING_RATE_DECAY)
    loss_func = nn.CrossEntropyLoss()

    def train():
        gcn.train()
        classifier.train()
        optimizer.zero_grad()

        embeddings_a = gcn.forward(features_a, adj_a)
        predict_a = classifier.forward(embeddings_a)
        loss = loss_func(predict_a, torch.argmax(labels_a, dim=1))
        loss.backward()
        optimizer.step()

    def evaluate():
        gcn.eval()
        classifier.eval()

        embeddings_a = gcn.forward(features_a, adj_a)
        predict_a = classifier.forward(embeddings_a)
        embeddings_b = gcn.forward(features_b, adj_b)
        predict_b = classifier.forward(embeddings_b)
        f1_aa = f1_score(torch.argmax(labels_a, dim=1), torch.argmax(predict_a, dim=1), average='macro')
        f1_ab = f1_score(torch.argmax(labels_b, dim=1), torch.argmax(predict_b, dim=1), average='macro')
        print(f'\tIn epoch {epoch}')
        print(f'\t\tA->A: {f1_aa}')
        print(f'\t\tA->B: {f1_ab}')

    t = tqdm(range(N_EPOCH), total=N_EPOCH) if use_tqdm else range(N_EPOCH)
    for i in t:
        epoch = i + 1
        scheduler.step(epoch=epoch)
        train()
        if epoch % N_EVAL == 0:
            evaluate()


def train_multi(start: str, use_tqdm=False):
    print(f'\tStart from graph {start}:')
    if start == 'A':
        src, tgt = 'chn', 'usa'
    elif start == 'B':
        src, tgt = 'usa', 'chn'
    else:
        assert False
    adj_a, features_a, labels_a = load_coauthor(src)
    adj_b, features_b, labels_b = load_coauthor(tgt)
    n_f = features_a.shape[1]
    n_l = labels_a.shape[1]
    adj_a = torch.from_numpy(adj_a).type(torch.float32)
    adj_b = torch.from_numpy(adj_b).type(torch.float32)
    features_a = torch.from_numpy(features_a).type(torch.float32)
    features_b = torch.from_numpy(features_b).type(torch.float32)
    labels_a = torch.from_numpy(labels_a).type(torch.float32)
    labels_b = torch.from_numpy(labels_b).type(torch.float32)

    gcn = GCN(n_f, 64, 64)
    classifier = Classifier(64, n_l)
    optimizer = optim.Adam(
        params=chain(gcn.parameters(), classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, N_EVAL, gamma=1 - LEARNING_RATE_DECAY)
    loss_func = nn.BCEWithLogitsLoss()

    def train():
        gcn.train()
        classifier.train()
        optimizer.zero_grad()

        embeddings_a = gcn.forward(features_a, adj_a)
        predict_a = classifier.forward(embeddings_a)
        loss = loss_func(predict_a, labels_a)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch=epoch)

    def evaluate():
        gcn.eval()
        classifier.eval()

        embeddings_a = gcn.forward(features_a, adj_a)
        predict_a = classifier.forward(embeddings_a)
        embeddings_b = gcn.forward(features_b, adj_b)
        predict_b = classifier.forward(embeddings_b)
        f1_aa, f1_ab = 0, 0
        for j in range(n_l):
            f1_aa += f1_score(labels_a[:, j].type(torch.int64),
                              (predict_a[:, j] > 0.5).type(torch.int64),
                              average='macro')
            f1_ab += f1_score(labels_b[:, j].type(torch.int64),
                              (predict_b[:, j] > 0.5).type(torch.int64),
                              average='macro')
        print(f'\tIn epoch {epoch}')
        print(f'\t\tA->A: {f1_aa / n_l}')
        print(f'\t\tA->B: {f1_ab / n_l}')

    t = tqdm(range(N_EPOCH), total=N_EPOCH) if use_tqdm else range(N_EPOCH)
    for i in t:
        epoch = i + 1
        train()
        if epoch % N_EVAL == 0:
            evaluate()


if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    print(f'For citation:')
    train_single('A')
    train_single('B')
    print(f'For coauthor:')
    train_multi('A')
    train_multi('B')
