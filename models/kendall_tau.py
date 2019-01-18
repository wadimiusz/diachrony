import gensim
from scipy.stats import mstats

from utils import log


def word_index(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors, word: str) -> object:
    """
    A handy function for extracting the word index from models
    :param w2v1: the model in question. if present, we use the index from that model
    :param w2v2: if word not present in w2v1, we look it up in the second model, w2v2
    :param word: word the index of which we extract
    :return: the index of the word, an integer
    """
    if word in w2v1.wv:
        return w2v1.wv.vocab[word].index
    else:
        return len(w2v1.wv.vocab) + w2v2.wv.vocab[word].index


def get_changes_by_kendalltau(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                              top_n_changed_words: int, top_n_neighbors):

    log('Doing kendall tau')
    result = list()
    for num, word in enumerate(w2v1.wv.vocab.keys()):
        if num % 10 == 0:
            log("{words_num} / {length}".format(words_num=num, length=len(w2v1.wv.vocab)), end='\r')

        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        if len(top_n_1) == len(top_n_2) == top_n_neighbors:
            top_n_1 = [word_index(w2v1=w2v1, w2v2=w2v2, word=word) for word in top_n_1]
            top_n_2 = [word_index(w2v1=w2v1, w2v2=w2v2, word=word) for word in top_n_2]
            score, p_value = mstats.kendalltau(top_n_1, top_n_2)
            result.append((word, score))

    result = sorted(result, key=lambda x: x[1], )[:top_n_changed_words]
    log('\nDONE')
    return result


if __name__ == '__main__':
    pass
