import numpy as np


def change(directory: str, name: str):
    with open(f'{directory}/{name}.emb') as fp:
        n, d = fp.readline().split()
        n, d = int(n), int(d)
        i_features = np.genfromtxt(fp)
    print(n, d)
    features = np.zeros(shape=[n, d], dtype=np.float)
    for i, *f in i_features:
        features[int(i), :] = f
    np.savetxt(f'{directory}/{name}.txt', features, fmt='%.10f')


if __name__ == '__main__':
    change('transfer', 'chn')
    change('transfer', 'usa')
    change('transfer3', 'chn')
    change('transfer3', 'usa')
