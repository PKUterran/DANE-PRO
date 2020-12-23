import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from reader import load_citation, load_coauthor

RANDOM_SEED = 0

# deepwalk --input data/transfer/chn.cites --output pro-data/transfer/chn.emb --format edgelist
# deepwalk --input data/transfer/usa.cites --output pro-data/transfer/usa.emb --format edgelist
# deepwalk --input data/transfer3/chn.cites --output pro-data/transfer3/chn.emb --format edgelist
# deepwalk --input data/transfer3/usa.cites --output pro-data/transfer3/usa.emb --format edgelist


def train_single():
    print('For citation:')
    _, _, labels_a = load_citation('chn')
    _, _, labels_b = load_citation('usa')
    labels_a, labels_b = np.argmax(labels_a, axis=1), np.argmax(labels_b, axis=1)
    features_a = np.genfromtxt('pro-data/transfer/chn.txt', dtype=np.float)
    features_b = np.genfromtxt('pro-data/transfer/usa.txt', dtype=np.float)

    classifier_a = SGDClassifier(loss='log', max_iter=1000, random_state=RANDOM_SEED)
    classifier_a.fit(features_a, labels_a)
    predict_aa = classifier_a.predict(features_a)
    predict_ab = classifier_a.predict(features_b)
    classifier_b = SGDClassifier(loss='log', max_iter=1000, random_state=RANDOM_SEED)
    classifier_b.fit(features_b, labels_b)
    predict_ba = classifier_b.predict(features_a)
    predict_bb = classifier_b.predict(features_b)
    f1_aa = f1_score(labels_a, predict_aa, average='macro')
    f1_ab = f1_score(labels_b, predict_ab, average='macro')
    f1_ba = f1_score(labels_a, predict_ba, average='macro')
    f1_bb = f1_score(labels_b, predict_bb, average='macro')
    print(f'\tA->A: {f1_aa}')
    print(f'\tA->B: {f1_ab}')
    print(f'\tB->A: {f1_ba}')
    print(f'\tB->B: {f1_bb}')


def train_multi():
    print('For coauthor:')
    _, _, labels_a = load_coauthor('chn')
    _, _, labels_b = load_coauthor('usa')
    features_a = np.genfromtxt('pro-data/transfer3/chn.txt', dtype=np.float)
    features_b = np.genfromtxt('pro-data/transfer3/usa.txt', dtype=np.float)

    classifier_a = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=1000, random_state=RANDOM_SEED))
    classifier_a.fit(features_a, labels_a)
    predict_aa = classifier_a.predict(features_a)
    predict_ab = classifier_a.predict(features_b)
    classifier_b = MultiOutputClassifier(SGDClassifier(loss='log', max_iter=1000, random_state=RANDOM_SEED))
    classifier_b.fit(features_b, labels_b)
    predict_ba = classifier_b.predict(features_a)
    predict_bb = classifier_b.predict(features_b)
    f1_aa, f1_ab, f1_ba, f1_bb = 0, 0, 0, 0
    for i in range(4):
        f1_aa += f1_score(labels_a[:, i], predict_aa[:, i], average='macro')
        f1_ab += f1_score(labels_b[:, i], predict_ab[:, i], average='macro')
        f1_ba += f1_score(labels_a[:, i], predict_ba[:, i], average='macro')
        f1_bb += f1_score(labels_b[:, i], predict_bb[:, i], average='macro')
    print(f'\tA->A: {f1_aa / 4}')
    print(f'\tA->B: {f1_ab / 4}')
    print(f'\tB->A: {f1_ba / 4}')
    print(f'\tB->B: {f1_bb / 4}')


if __name__ == '__main__':
    train_single()
    train_multi()
