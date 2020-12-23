import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import chain
from sklearn.metrics import f1_score

from reader import load_citation, load_coauthor
from net.gcn import GCN, Classifier
from net.dis import Discriminator

LEARNING_RATE = 1e-3
LEARNING_RATE_D = 1e-2
WEIGHT_DECAY = 5e-4
LAMBDA_SINGLE = 5
LAMBDA_MULTI = 1 / 8
N_EPOCH = 200
N_EVAL = 1
N_DIS_STEP = 5
N_DIS_BATCH = 256
RANDOM_SEED = 0


def set_seed(seed):
    torch.manual_seed(seed)


def train_gcn(single: bool, start: str, use_tqdm=False):
    print(f'\tStart from graph {start}:')
    if start == 'A':
        src, tgt = 'chn', 'usa'
    elif start == 'B':
        src, tgt = 'usa', 'chn'
    else:
        assert False
    adj_a, features_a, labels_a = load_citation(src) if single else load_coauthor(src)
    adj_b, features_b, labels_b = load_citation(tgt) if single else load_coauthor(tgt)
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
    discriminator = Discriminator(64, 64, p_dropout=0.5)
    optimizer_g = optim.Adam(
        params=chain(gcn.parameters(), classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    optimizer_d = optim.Adam(
        params=discriminator.parameters(),
        lr=LEARNING_RATE_D
    )
    loss_func = nn.CrossEntropyLoss() if single else nn.BCEWithLogitsLoss()
    bce_func = nn.BCEWithLogitsLoss()

    def dis_loss(src_emb: torch.Tensor, tgt_emb: torch.Tensor) -> torch.Tensor:
        src_sample = torch.randint(high=src_emb.shape[0], size=[N_DIS_BATCH])
        tgt_sample = torch.randint(high=tgt_emb.shape[0], size=[N_DIS_BATCH])
        src_emb = src_emb[src_sample, :]
        tgt_emb = tgt_emb[tgt_sample, :]
        src_der = discriminator.forward(src_emb)
        tgt_der = discriminator.forward(tgt_emb)
        der = torch.cat([src_der, tgt_der], dim=0)
        der_t = torch.cat([torch.zeros_like(src_der), torch.ones_like(tgt_der)], dim=0)
        return bce_func(der, der_t)

    def train_d():
        gcn.eval()
        classifier.eval()
        discriminator.train()
        optimizer_d.zero_grad()

        embeddings_a = gcn.forward(features_a, adj_a)
        embeddings_b = gcn.forward(features_b, adj_b)
        loss = dis_loss(embeddings_a, embeddings_b)
        loss.backward()
        optimizer_d.step()

    def train_g():
        gcn.train()
        classifier.train()
        discriminator.eval()
        optimizer_g.zero_grad()

        embeddings_a = gcn.forward(features_a, adj_a)
        embeddings_b = gcn.forward(features_b, adj_b)
        predict_a = classifier.forward(embeddings_a)
        loss_g = loss_func(predict_a, torch.argmax(labels_a, dim=1)) if single else loss_func(predict_a, labels_a)
        loss_d = dis_loss(embeddings_b, embeddings_a)
        loss = loss_g + LAMBDA_SINGLE * loss_d if single else loss_g + LAMBDA_MULTI * loss_d
        # print(loss_g.item(), loss_d.item())
        loss.backward()
        optimizer_g.step()

    def evaluate():
        gcn.eval()
        classifier.eval()

        embeddings_a = gcn.forward(features_a, adj_a)
        predict_a = classifier.forward(embeddings_a)
        embeddings_b = gcn.forward(features_b, adj_b)
        predict_b = classifier.forward(embeddings_b)
        if single:
            f1_aa = f1_score(torch.argmax(labels_a, dim=1), torch.argmax(predict_a, dim=1), average='macro')
            f1_ab = f1_score(torch.argmax(labels_b, dim=1), torch.argmax(predict_b, dim=1), average='macro')
            print(f'\tIn epoch {epoch}')
            print(f'\t\tA->A: {f1_aa}')
            print(f'\t\tA->B: {f1_ab}')
        else:
            f1_aa, f1_ab = 0, 0
            for j in range(n_l):
                f1_aa += f1_score(labels_a[:, j].type(torch.int64),
                                  (torch.sigmoid(predict_a[:, j]) > 0.5).type(torch.int64),
                                  average='macro')
                f1_ab += f1_score(labels_b[:, j].type(torch.int64),
                                  (torch.sigmoid(predict_b[:, j]) > 0.5).type(torch.int64),
                                  average='macro')
            print(f'\tIn epoch {epoch}')
            print(f'\t\tA->A: {f1_aa / n_l}')
            print(f'\t\tA->B: {f1_ab / n_l}')

    t = tqdm(range(N_EPOCH), total=N_EPOCH) if use_tqdm else range(N_EPOCH)
    for i in t:
        epoch = i + 1
        for _ in range(N_DIS_STEP):
            train_d()
        train_g()
        if epoch % N_EVAL == 0:
            evaluate()


if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    print(f'For citation:')
    train_gcn(True, 'A')
    train_gcn(True, 'B')
    print(f'For coauthor:')
    train_gcn(False, 'A')
    train_gcn(False, 'B')
