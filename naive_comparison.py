import argparse
import gensim
import os


def comparison(w2v1_path: str, w2v2_path: str, binary1: bool, binary2: bool, top_n_neighbors: int,
               top_n_changed_words: (int, None), top_n_most_frequent_words: (int, None), pos_tag: str,
               verbose: bool):
    """
    This module extracts two models from two specified paths and compares the meanings of words within their vocabulary.
    :param w2v1_path: the path to the first model
    :param w2v2_path: the path to the second model
    :param binary1: if True, open the first model as binary
    :param binary2: if True, open the second model as binary
    :param top_n_neighbors: we will compare top n neighbors of words
    :param top_n_changed_words: we will output top n most interesting words, may be int or None
    :param top_n_most_frequent_words: we will use top n most frequent words from each model, may be int or None
    :param pos_tag: specify this to consider only words with a specific pos_tag
    :param verbose: if True the model is verbose (gives system messages)
    :return: None
    """

    if not os.path.exists(w2v1_path):
        raise FileNotFoundError("File {path} is not found".format(path=w2v1_path))

    if not os.path.exists(w2v2_path):
        raise FileNotFoundError("File {path} is not found".format(path=w2v2_path))

    if verbose:
        print("Loading the first model")
    w2v1 = gensim.models.KeyedVectors.load_word2vec_format(w2v1_path, binary=binary1)
    if verbose:
        print("Success.")
        print("Loading the second model")
    w2v2 = gensim.models.KeyedVectors.load_word2vec_format(w2v2_path, binary=binary2)
    if verbose:
        print("Success.")

    vocab1 = list(w2v1.vocab.keys())
    vocab2 = list(w2v2.vocab.keys())

    if pos_tag is not None:
        vocab1 = [word for word in vocab1 if word.endswith(pos_tag)]
        vocab2 = [word for word in vocab2 if word.endswith(pos_tag)]

    vocab1 = vocab1[:top_n_most_frequent_words]
    vocab2 = vocab2[:top_n_most_frequent_words]

    shared_vocabulary = [word for word in vocab1 if word in vocab2]

    n = top_n_neighbors

    results = list()
    for word in shared_vocabulary:
        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=n)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=n)]
        intersection = [word for word in top_n_1 if word in top_n_2]
        union = set(top_n_1 + top_n_2)
        jaccard = len(intersection) / len(union)
        results.append((word, jaccard))

    results = sorted(results, key=lambda x: x[1])[:top_n_changed_words]
    for word, jaccard in results:
        print("word {word} has jaccard measure {jaccard}".format(word=word, jaccard=jaccard))
        print("word {word} has the following neighbors in model1:")
        print(*[word for word, score in w2v1.most_similar(word)], sep=",")
        print("word {word} has the following neighbors in model2:")
        print(*[word for word, score in w2v2.most_similar(word)], sep=",")
        print("==========================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, help="the path to the first model")
    parser.add_argument('--binary1', action="store_true", help="If this argument if given, the first model"
                                                               " is read is binary")
    parser.add_argument('--model2', type=str, help="the path to the first model")
    parser.add_argument('--binary2', action="store_true", help="If this argument if given, the second model"
                                                               " is read is binary")
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

    args = parser.parse_args()
    comparison(w2v1_path=args.model1, w2v2_path=args.model2, binary1=args.binary1, binary2=args.binary2,
               top_n_neighbors=args.top_n_neighbors, top_n_most_frequent_words=args.top_n_most_frequent_words,
               pos_tag=args.pos_tag, verbose=args.verbose, top_n_changed_words=args.top_n_changed_words)


if __name__ == "__main__":
    main()
