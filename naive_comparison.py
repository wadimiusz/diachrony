import argparse
import gensim
import os
import random
from scipy.stats import mstats


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


def tag_cutter(word: str):
    """
    :param word: a string with our without a tag, e g дом_NOUN or дом
    :return: the word without tags, e. g. дом
    """

    if "_" in word:
        return word[:word.find("_")]
    else:
        return word


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

    if verbose:
        print("Loading the first model")
    w2v1 = load_model(w2v1_path)
    if verbose:
        print("Success.")
        print("Loading the second model")
    w2v2 = load_model(w2v2_path)
    if verbose:
        print("Success.")

    return w2v1, w2v2


def get_top_neighbors(word: str, w2v: gensim.models.KeyedVectors, vocab: list, n: int, cut_tags: bool):
        """
        :param word: the word neighbors of which to retrieve
        :param w2v: the keyed vectors model
        :param vocab: we only retrieve words that are included in vocab
        :param n: top n neighbors will be extracted
        :param cut_tags: if True: i. g. word="дом", it will retrieve all instances of дом with tags: дом_NOUN etc.
        :return: n closest neighbors for the given word (for each tag if cut_tags is True, so if there is word_NOUN,
                word_ADJ and word_VERB for the same word, it returns 30 neighbors
        """
        if cut_tags:
            result = list()
            for word in vocab:
                if word.startswith(word + "_"):
                    result.extend([word for word, score in w2v.most_similar(word, topn=n*5)])
        else:
            result = [word for word, score in w2v.most_similar(word, topn=n*5)]

        result = [word for word in result if word in vocab][:n]
        return result


def preprocess_vocabs(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors, vocab1: list, vocab2: list,
                      pos_tag: str, verbose: bool, cut_tags: bool, top_n_most_frequent_words: int):
    """
    This function does the preprocessing and cleanup of embedding vocabs before we can do the main job
    :param w2v1: the first model the vocabulary of which we preprocess. we use the model to extract the frequencies
    :param w2v2: the second model the vocabulary of which we preprocess. we use the model to extract the frequencies
    :param vocab1: list
    :param vocab2: list
    :param pos_tag: str or None, if str we only look for words with this pos tag
    :param verbose: bool, if True we print system messages
    :param cut_tags: bool, if True we ignore pos tags
    :param top_n_most_frequent_words:
    :return: vocab1, vocab2, both are preprocessed lists
    """
    if verbose:
        print("Preprocessing...")

    vocab1 = sorted(vocab1, key=lambda x: word_frequency(w2v1, x), reverse=True)
    vocab2 = sorted(vocab2, key=lambda x: word_frequency(w2v2, x), reverse=True)

    if cut_tags:
        vocab1 = [tag_cutter(word) for word in vocab1]
        vocab2 = [tag_cutter(word) for word in vocab2]

    if pos_tag is not None:
        vocab1 = [word for word in vocab1 if word.endswith(pos_tag)]
        vocab2 = [word for word in vocab2 if word.endswith(pos_tag)]

    vocab1 = vocab1[:top_n_most_frequent_words]
    vocab2 = vocab2[:top_n_most_frequent_words]

    return vocab1, vocab2


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


def get_changes_by_jaccard(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                           vocab1: list, vocab2: list, top_n_neighbors: int, cut_tags: bool,
                           verbose: bool, top_n_changed_words: int, shared_vocabulary: list):
    if verbose:
        print("JACCARD")

    results = list()
    for num, word in enumerate(shared_vocabulary):
        if verbose and num % 10 == 0:
            print("{words_num} / {length}".format(words_num=num, length=len(shared_vocabulary)), end='\r')

        top_n_1 = get_top_neighbors(word=word, w2v=w2v1, vocab=vocab1, n=top_n_neighbors, cut_tags=cut_tags)
        top_n_2 = get_top_neighbors(word=word, w2v=w2v2, vocab=vocab2, n=top_n_neighbors, cut_tags=cut_tags)
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
        top_n_1 = get_top_neighbors(word=word, w2v=w2v1, vocab=vocab1, n=top_n_neighbors, cut_tags=cut_tags)
        top_n_2 = get_top_neighbors(word=word, w2v=w2v2, vocab=vocab2, n=top_n_neighbors, cut_tags=cut_tags)
        print("word {word} has jaccard measure {jaccard}".format(word=word, jaccard=jaccard))
        print("word {word} has the following neighbors in model1:".format(word=word))
        print(*top_n_1, sep=',')
        print("word {word} has the following neighbors in model2:".format(word=word))
        print(*top_n_2, sep=',')
        print("==========================================================")


def get_changes_by_kendalltau(w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                              vocab1: list, vocab2: list, top_n_neighbors: int, cut_tags: bool,
                              verbose: bool, top_n_changed_words: int, shared_vocabulary: list):

    if verbose:
        print("KENDALL TAU")
    result = list()
    for num, word in enumerate(shared_vocabulary):
        if verbose and num % 10 == 0:
            print("{words_num} / {length}".format(words_num=num, length=len(shared_vocabulary)), end='\r')

        top_n_1 = get_top_neighbors(word=word, w2v=w2v1, vocab=vocab1, n=top_n_neighbors, cut_tags=cut_tags)
        top_n_2 = get_top_neighbors(word=word, w2v=w2v2, vocab=vocab2, n=top_n_neighbors, cut_tags=cut_tags)
        if len(top_n_1) == len(top_n_2) == top_n_neighbors:
            top_n_1 = [word_index(w2v1=w2v1, w2v2=w2v2, word=word) for word in top_n_1]
            top_n_2 = [word_index(w2v1=w2v1, w2v2=w2v2, word=word) for word in top_n_2]
            score, p_value = mstats.kendalltau(top_n_1, top_n_2)
            result.append((word, score))

    result = sorted(result, key=lambda x: x[1], )[:top_n_changed_words]
    for word, score in result:
            top_n_1 = get_top_neighbors(word=word, w2v=w2v1, vocab=vocab1, n=top_n_neighbors, cut_tags=cut_tags)
            top_n_2 = get_top_neighbors(word=word, w2v=w2v2, vocab=vocab2, n=top_n_neighbors, cut_tags=cut_tags)

            print("word {word} has kendall-tau score {score}".format(word=word, score=score))
            print("word {word} has the following neighbors in model1:".format(word=word))
            print(*top_n_1, sep=',')
            print("word {word} has the following neighbors in model2:".format(word=word))
            print(*top_n_2, sep=',')
            print("==========================================================")


def comparison(w2v1_path: str, w2v2_path: str, top_n_neighbors: int,
               top_n_changed_words: (int, None), top_n_most_frequent_words: (int, None), pos_tag: (str, None),
               verbose: bool, cut_tags: bool):
    """
    This module extracts two models from two specified paths and compares the meanings of words within their vocabulary.
    :param w2v1_path: the path to the first model
    :param w2v2_path: the path to the second modet
    :param top_n_neighbors: we will compare top n neighbors of words
    :param top_n_changed_words: we will output top n most interesting words, may be int or None
    :param top_n_most_frequent_words: we will use top n most frequent words from each model, may be int or None
    :param pos_tag: specify this to consider only words with a specific pos_tag
    :param verbose: if True the model is verbose (gives system messages)
    :param cut_tags: if True words like дом_NOUN turn into words like дом, without tags
    :return: None
    """
    w2v1, w2v2 = loader(w2v1_path=w2v1_path, w2v2_path=w2v2_path, verbose=verbose)

    vocab1 = list(w2v1.vocab.keys())
    vocab2 = list(w2v2.vocab.keys())

    if verbose:
        print("The first model contains {words1} words, e. g. {word1}\n"
              "The second model contains {words2} words, e. g. {word2}".format(words1=len(vocab1), words2=len(vocab2),
                                                                               word1=random.choice(vocab1),
                                                                               word2=random.choice(vocab2)))

    vocab1, vocab2 = preprocess_vocabs(w2v1=w2v1, w2v2=w2v2, vocab1=vocab1, vocab2=vocab2, pos_tag=pos_tag,
                                       verbose=verbose, cut_tags=cut_tags,
                                       top_n_most_frequent_words=top_n_most_frequent_words)

    if verbose:
        print("After preprocessing, the first model contains {words1} words, e. g. {word1}\n"
              "The second model contains {words2} words, e. g. {word2}".format(words1=len(vocab1), words2=len(vocab2),
                                                                               word1=random.choice(vocab1),
                                                                               word2=random.choice(vocab2)))

    shared_vocabulary = list(set(vocab1).intersection(set(vocab2)))
    if verbose:
        print("The shared vocabulary contains {n} words".format(n=len(shared_vocabulary)))
    get_changes_by_jaccard(w2v1=w2v1, w2v2=w2v2, vocab1=vocab1, vocab2=vocab2, top_n_neighbors=top_n_neighbors,
                           cut_tags=cut_tags, top_n_changed_words=top_n_changed_words, verbose=verbose,
                           shared_vocabulary=shared_vocabulary)

    get_changes_by_kendalltau(w2v1=w2v1, w2v2=w2v2, vocab1=vocab1, vocab2=vocab2, top_n_neighbors=top_n_neighbors,
                              cut_tags=cut_tags, top_n_changed_words=top_n_changed_words, verbose=verbose,
                              shared_vocabulary=shared_vocabulary)


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

    parser.add_argument("--cut-tags", action="store_true", help="Use this argument to cut off the pos-tags, e. g. "
                                                                "дом_NOUN --> дом")

    args = parser.parse_args()
    comparison(w2v1_path=args.model1, w2v2_path=args.model2, top_n_neighbors=args.top_n_neighbors,
               top_n_most_frequent_words=args.top_n_most_frequent_words, pos_tag=args.pos_tag, verbose=args.verbose,
               top_n_changed_words=args.top_n_changed_words, cut_tags=args.cut_tags)


if __name__ == "__main__":
    main()
