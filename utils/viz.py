import sys
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils import load_model, intersection_align_gensim


def sims_aligned(year, word, *args):
    all_most_sim = []
    wrd_vectors = []

    def unite_sims(m):
        for x in m:
            all_most_sim.append(x[0])

    for pair in combinations(args, 2):
        _, _ = intersection_align_gensim(pair[0], pair[1])

    for i in range(len(args)):
        unite_sims(args[i].most_similar(word, topn=7))
        if i != 0:
            wrd_vectors.append(args[i][word])

    return all_most_sim, wrd_vectors, args[0], word, year


def viz(all_most_sim, wrd_vectors, model, word, year):
    rows = len(wrd_vectors) + len(all_most_sim) + 1
    arr = np.empty((rows, model.vector_size), dtype='f')
    word_labels = [word + ' ' + year]

    num = int(year) - 1

    arr[0, :] = model[word]

    row_counter = 1

    for i in range(len(wrd_vectors)):
        arr[row_counter, :] = wrd_vectors[i]
        row_counter += 1
        word_labels.append(word + ' ' + str(num))
        num -= 1

    for i in all_most_sim:
        wrd_vector = model[i]
        word_labels.append(i)
        arr[row_counter, :] = wrd_vector

    print(arr)
    print(arr.shape)
    tsne = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
    np.set_printoptions(suppress=True)
    embedded = tsne.fit_transform(arr)

    wrd_coord = []

    for i in range(0, len(wrd_vectors) + 1):
        a = embedded[i]
        wrd_coord.append(a)

    x_coords = embedded[:, 0]
    y_coords = embedded[:, 1]

    plt.scatter(x_coords, y_coords)
    plt.axis('off')

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() - 10, x_coords.max() + 10)
    plt.ylim(y_coords.min() - 10, y_coords.max() + 10)
    for i in range(len(wrd_coord) - 1, 0, -1):
        plt.annotate("", xy=(wrd_coord[i - 1][0], wrd_coord[i - 1][1]),
                     xytext=(wrd_coord[i][0], wrd_coord[i][1]),
                     arrowprops=dict(arrowstyle='-|>', color='indianred'))

    plt.show()


def main():
    all_most_sim, wrd_vectors, model, word, year = sims_aligned('2002', 'железный_ADJ',
                                                                load_model(sys.argv[1]),
                                                                load_model(sys.argv[2]),
                                                                load_model(sys.argv[3]))

    viz(all_most_sim, wrd_vectors, model, word, year)


if __name__ == '__main__':
    main()
