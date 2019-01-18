import gensim

from utils import log


def get_global_anchors(word: str, w2v: gensim.models.KeyedVectors):
    """
    This takes in a word and a KeyedVectors model and returns a vector of cosine distances between this word and each
    word in the vocab.
    :param word:
    :param w2v:
    :return: np.array of distances shaped (len(w2v.vocab),)
    """
    word_vector = w2v.get_vector(word)
    return gensim.models.KeyedVectors.cosine_similarities(word_vector, w2v.vectors)


def get_changes_by_global_anchors(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                                  top_n_changed_words: int):
    """
    This method uses approach described in
    Yin, Zi, Vin Sachidananda, and Balaji Prabhakar. "The global anchor method for quantifying linguistic shifts and
    domain adaptation." Advances in Neural Information Processing Systems. 2018.
    It can be described as follows. To evaluate how much the meaning of a given word differs in two given corpora,
    we take cosine distance from the given word to all words in the vocabulary; those values make up a vector wich
    as much components as there are words in the vocab. We do it for both corpora and then compute the cosine distance
    between those two vectors
    :param w2v1: the first model
    :param w2v2: the second model, must have the same vocab as the first
    :param top_n_changed_words: we will output n words that differ the most in the given corpora
    :return: list of pairs (word, score), where score indicates how much a word has changed
    """
    log('Doing global anchors')
    result = list()
    for num, word in enumerate(w2v1.wv.vocab.keys()):
        if num % 10 == 0:
            log("{num} / {length}".format(num=num, length=len(w2v1.wv.vocab)), end='\r')

        w2v1_anchors = get_global_anchors(word, w2v1)
        w2v2_anchors = get_global_anchors(word, w2v2)

        score = gensim.models.KeyedVectors.cosine_similarities(w2v1_anchors, w2v2_anchors.reshape(1, -1))[0]
        result.append((word, score))

    result = sorted(result, key=lambda x: x[1])
    result = result[:top_n_changed_words]
    log('\nDone')
    return result
