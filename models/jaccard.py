import gensim

from utils import log


def get_changes_by_jaccard(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors, top_n_neighbors: int,
                           top_n_changed_words: int):
    log('Doing jaccard')
    result = list()
    for num, word in enumerate(w2v1.wv.vocab):
        if num % 10 == 0:
            log("{words_num} / {length}".format(words_num=num, length=len(w2v1.wv.vocab.keys())), end='\r')

        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        if len(top_n_1) == top_n_neighbors and len(top_n_2) == top_n_neighbors:
            intersection = set(top_n_1).intersection(set(top_n_2))
            union = set(top_n_1 + top_n_2)
            jaccard = len(intersection) / len(union)
            result.append((word, jaccard))

    result = sorted(result, key=lambda x: x[1], )[:top_n_changed_words]
    log('\nDone')
    return result


if __name__ == '__main__':
    pass
