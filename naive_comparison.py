import argparse
import gensim
import os


def comparison(w2v1_path: str, w2v2_path: str, binary1, binary2):
    """
    This module extracts two models from two specified paths and compares the meanings of words within their vocaublary.
    :param w2v1_path: the path to the first model
    :param w2v2_path: the path to the second model
    :param binary1: if True, open the first model as binary
    :param binary2: if True, open the second model as binary
    :return:
    """

    if not os.path.exists(w2v1_path):
        raise FileNotFoundError("File {path} is not found".format(path=w2v1_path))

    if not os.path.exists(w2v2_path):
        raise FileNotFoundError("File {path} is not found".format(path=w2v2_path))

    print("Loading the first model")
    w2v1 = gensim.models.KeyedVectors.load_word2vec_format(w2v1_path, binary=binary1)
    print("Loading the second model")
    w2v2 = gensim.models.KeyedVectors.load_word2vec_format(w2v2_path, binary=binary2)

    vocab1 = list(w2v1.vocab.keys())[:1000]
    vocab2 = list(w2v2.vocab.keys())[:1000]

    shared_vocabulary = [word for word in vocab1 if word in vocab2]

    n = 100  # потом вставить как параметр

    results = list()
    for word in shared_vocabulary:
        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=n)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=n)]
        intersection = [word for word in top_n_1 if word in top_n_2]
        union = set(top_n_1 + top_n_2)
        jaccard = len(intersection) / len(union)
        results.append((word, jaccard))

    results = sorted(results, key=lambda x: x[1])
    for word, jaccard in results:
        print("word {word} has jaccard measure {jaccard}".format(word=word, jaccard=jaccard))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, help="the path to the first model")
    parser.add_argument('--binary1', action="store_true", help="If this argument if given, the first model"
                                                               " is read is binary")
    parser.add_argument('--model2', type=str, help="the path to the first model")
    parser.add_argument('--binary2', action="store_true", help="If this argument if given, the second model"
                                                               " is read is binary")
    args = parser.parse_args()
    comparison(w2v1_path=args.model1, w2v2_path=args.model2, binary1=args.binary1, binary2=args.binary2)


if __name__ == "__main__":
    main()
