import argparse
import gensim
import os
import sys
import random
import numpy as np
from scipy.stats import mstats


def log(message: str, verbose: bool, end: str = '\n'):
    if verbose:
        sys.stderr.write(message+end)
        sys.stderr.flush()


def load_model(embeddings_file):
    """
    This function, written by github.com/akutuzov, unifies various standards of word embedding files.
    It automatically determines the format by the file extension and loads it from the disk correspondingly.
    :param embeddings_file: path to the file
    :return: the loaded model
    """
    # Determine the model format by the file extension
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Binary word2vec file
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Text word2vec file
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)
    return emb_model


# def tag_cutter(word: str):
#    """
#    :param word: a string with our without a tag, e g дом_NOUN or дом
#    :return: the word without tags, e. g. дом
#    """
#
#    if "_" in word:
#        return word[:word.find("_")]
#    else:
#        return word


def intersection_align_gensim(m1: gensim.models.KeyedVectors, m2: gensim.models.KeyedVectors,
                              pos_tag: (str, None) = None, words: (list, None) = None,
                              top_n_most_frequent_words: (int, None) = None):
    """
    This procedure, taken from https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf and slightly
    modified, corrects two models in a way that only the shared words of the vocabulary are kept in the model,
    and both vocabularies are sorted by frequencies.
    Original comment is as follows:

    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.

    :param m1: the first model
    :param m2: the second model
    :param pos_tag: if given, we remove words with other pos tags
    :param words: a container
    :param top_n_most_frequent_words: if not None, we only use top n words by frequency
    :return m1, m2: both models after their vocabs are modified
    """
    
    # Get the vocab for each model
    if pos_tag is None:
        vocab_m1 = set(m1.wv.vocab.keys())
        vocab_m2 = set(m2.wv.vocab.keys())
    else:
        vocab_m1 = set(word for word in m1.wv.vocab.keys() if word.endswith("_" + pos_tag))
        vocab_m2 = set(word for word in m2.wv.vocab.keys() if word.endswith("_" + pos_tag))

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words:
        common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab and top_n_most_frequent_words is not None:
        return m1, m2

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count, reverse=True)
    common_vocab = common_vocab[:top_n_most_frequent_words]
    
    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.syn0norm = m.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = dict()
        for new_index, word in enumerate(common_vocab):
            old_vocab_obj = old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return m1, m2


def loader(w2v1_path: str, w2v2_path: str, verbose: bool):
    """
    This function downloads two models using load_model and, optionally, reports success.
    :param w2v1_path: path to the first model
    :param w2v2_path: path to the second model
    :param verbose: defines verbosity
    :return:
    """
    if not os.path.exists(w2v1_path):
        raise FileNotFoundError("File {path} is not found".format(path=w2v1_path))

    if not os.path.exists(w2v2_path):
        raise FileNotFoundError("File {path} is not found".format(path=w2v2_path))

    log("Loading the first model", verbose)
    w2v1 = load_model(w2v1_path)
    log("Success.\n Loading the second model", verbose)
    w2v2 = load_model(w2v2_path)
    log("Success.", verbose)

    return w2v1, w2v2


def word_frequency(model: gensim.models.KeyedVectors, word: str) -> int:
    """
    A handy function for extracting the word frequency from models
    :param model: the model in question
    :param word: extract the frequency of this word
    :return:
    """
    return model.wv.vocab[word].count


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


def get_changes_by_jaccard(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors, top_n_neighbors: int,
                           verbose: bool, top_n_changed_words: int):
    print("\n" + "JACCARD MEASURE".center(40, '='))

    results = list()
    for num, word in enumerate(w2v1.wv.vocab):
        if num % 10 == 0:
            log("{words_num} / {length}".format(words_num=num, length=len(w2v1.wv.vocab.keys())), verbose, end='\r')

        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        if len(top_n_1) == top_n_neighbors and len(top_n_2) == top_n_neighbors:
            intersection = set(top_n_1).intersection(set(top_n_2))
            union = set(top_n_1 + top_n_2)
            jaccard = len(intersection) / len(union)
            results.append((word, jaccard))
            # print("Intersection", *intersection)
            # print("Union", *union)
            # print("Intersection length", len(intersection))
            # print("Union length", len(union))

    results = sorted(results, key=lambda x: x[1], )[:top_n_changed_words]
    for word, jaccard in results:
        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        print("word {word} has jaccard measure {jaccard}".format(word=word, jaccard=jaccard))
        print("word {word} has the following neighbors in model1:".format(word=word))
        print(*top_n_1, sep=',')
        print("word {word} has the following neighbors in model2:".format(word=word))
        print(*top_n_2, sep=',')
        print("")


def get_changes_by_kendalltau(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors, top_n_neighbors: int,
                              verbose: bool, top_n_changed_words: int):
    print("\n" + "KENDALL TAU".center(40, '='))

    result = list()
    for num, word in enumerate(w2v1.wv.vocab.keys()):
        if num % 10 == 0:
            log("{words_num} / {length}".format(words_num=num, length=len(w2v1.wv.vocab)), verbose, end='\r')

        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        if len(top_n_1) == len(top_n_2) == top_n_neighbors:
            top_n_1 = [word_index(w2v1=w2v1, w2v2=w2v2, word=word) for word in top_n_1]
            top_n_2 = [word_index(w2v1=w2v1, w2v2=w2v2, word=word) for word in top_n_2]
            score, p_value = mstats.kendalltau(top_n_1, top_n_2)
            result.append((word, score))

    result = sorted(result, key=lambda x: x[1], )[:top_n_changed_words]
    for word, score in result:
        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]

        print("word {word} has kendall-tau score {score}".format(word=word, score=score))
        print("word {word} has the following neighbors in model1:".format(word=word))
        print(*top_n_1, sep=',')
        print("word {word} has the following neighbors in model2:".format(word=word))
        print(*top_n_2, sep=',')
        print("")


def comparison(w2v1_path: str, w2v2_path: str, top_n_neighbors: int,
               top_n_changed_words: (int, None), top_n_most_frequent_words: (int, None), pos_tag: (str, None),
               verbose: bool):
    """
    This module extracts two models from two specified paths and compares the meanings of words within their vocabulary.
    :param w2v1_path: the path to the first model
    :param w2v2_path: the path to the second modet
    :param top_n_neighbors: we will compare top n neighbors of words
    :param top_n_changed_words: we will output top n most interesting words, may be int or None
    :param top_n_most_frequent_words: we will use top n most frequent words from each model, may be int or None
    :param pos_tag: specify this to consider only words with a specific pos_tag
    :param verbose: if True the model is verbose (gives system messages)
    :return: None
    """
    w2v1, w2v2 = loader(w2v1_path=w2v1_path, w2v2_path=w2v2_path, verbose=verbose)

    log("The first model contains {words1} words, e. g. {word1}\n"
        "The second model contains {words2} words, e. g. {word2}".format(
               words1=len(w2v1.wv.vocab), words2=len(w2v2.wv.vocab),word1=random.choice(list(w2v1.wv.vocab.keys())),
               word2=random.choice(list(w2v2.wv.vocab.keys()))), verbose)

    w2v1, w2v2 = intersection_align_gensim(w2v1, w2v2, pos_tag=pos_tag,
                                           top_n_most_frequent_words=top_n_most_frequent_words)

    log("After preprocessing, the first model contains {words1} words, e. g. {word1}\n"
        "The second model contains {words2} words, e. g. {word2}".format(
               words1=len(w2v1.wv.vocab), words2=len(w2v2.wv.vocab),word1=random.choice(list(w2v1.wv.vocab.keys())),
               word2=random.choice(list(w2v2.wv.vocab.keys()))), verbose)

    get_changes_by_jaccard(w2v1=w2v1, w2v2=w2v2, top_n_neighbors=top_n_neighbors,
                           top_n_changed_words=top_n_changed_words, verbose=verbose)

    get_changes_by_kendalltau(w2v1=w2v1, w2v2=w2v2, top_n_neighbors=top_n_neighbors,
                              top_n_changed_words=top_n_changed_words, verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, help="the path to the first model", required=True)
    parser.add_argument('--model2', type=str, help="the path to the first model", required=True)
    parser.add_argument("--top-n-neighbors", type=int, default=10, help="We will compare top n nearest neighbors",
                        dest="top_n_neighbors")

    parser.add_argument("--top-n-changed-words", type=int, default=None, help="We will output top n changed words",
                        dest="top_n_changed_words")

    parser.add_argument("--top-n-most-frequent-words", type=int, default=None,
                        help="We will output top n changed words", dest="top_n_most_frequent_words")

    parser.add_argument("--pos-tag", type=str, default=None, help="If you want to see words with a specific "
                                                                  "pos tag, specify this", dest="pos_tag")

    parser.add_argument("--verbose", action="store_true", help="Give this argument "
                                                               "to make the model verbose")

    # parser.add_argument("--cut-tags", action="store_true", help="Use this argument to cut off the pos-tags, e. g. "
    #                                                            "дом_NOUN --> дом")

    args = parser.parse_args()
    comparison(w2v1_path=args.model1, w2v2_path=args.model2, top_n_neighbors=args.top_n_neighbors,
               top_n_most_frequent_words=args.top_n_most_frequent_words, pos_tag=args.pos_tag, verbose=args.verbose,
               top_n_changed_words=args.top_n_changed_words)


if __name__ == "__main__":
    main()
