import argparse
import gensim
import os
import random


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


def get_top_neighbors(word: str, w2v: gensim.models.KeyedVectors, n: int, cut_tags: bool):
        """
        :param word: the word whose neighbors to retrieve
        :param w2v: the keyed vectors model
        :param n: top n neighbors will be extracted
        :param cut_tags: if True: i. g. word="дом", it will retrieve all instances of дом with tags: дом_NOUN etc.
        :return: n closest neighbors for the given word (for each tag if cut_tags is True, so if there is word_NOUN,
                word_ADJ and word_VERB for the same word, it returns 30 neighbors
        """
        if cut_tags:
            result = list()
            for word in w2v.vocab:
                if word.startswith(word + "_"):
                    result.extend([word for word, score in w2v.most_similar(word, topn=n)])
            return result
        else:
            return [word for word, score in w2v.most_similar(word, topn=n)]


def comparison(w2v1_path: str, w2v2_path: str, top_n_neighbors: int,
               top_n_changed_words: (int, None), top_n_most_frequent_words: (int, None), pos_tag: str,
               verbose: bool, cut_tags: bool):
    """
    This module extracts two models from two specified paths and compares the meanings of words within their vocabulary.
    :param w2v1_path: the path to the first model
    :param w2v2_path: the path to the second model
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

    if cut_tags:
        vocab1 = [tag_cutter(word) for word in vocab1]
        vocab2 = [tag_cutter(word) for word in vocab2]

    if pos_tag is not None:
        vocab1 = [word for word in vocab1 if word.endswith(pos_tag)]
        vocab2 = [word for word in vocab2 if word.endswith(pos_tag)]

    vocab1 = vocab1[:top_n_most_frequent_words]
    vocab2 = vocab2[:top_n_most_frequent_words]

    shared_vocabulary = list(set(vocab1).intersection(set(vocab2)))
    if verbose:
        print("The shared vocabulary contains {n} words".format(n=len(shared_vocabulary)))

    n = top_n_neighbors

    results = list()
    for num, word in enumerate(shared_vocabulary):
        if verbose and num % 10 == 0:
            print("{words_num} / {length}".format(words_num=num, length=len(shared_vocabulary)), end='\r')

        top_n_1 = get_top_neighbors(word, w2v1, n, cut_tags)
        top_n_2 = get_top_neighbors(word, w2v2, n, cut_tags)
        if len(top_n_1) + len(top_n_2) != 0:
            intersection = [word for word in top_n_1 if word in top_n_2]
            union = set(top_n_1 + top_n_2)
            jaccard = len(intersection) / len(union)
            results.append((word, jaccard))

    results = sorted(results, key=lambda x: x[1])[:top_n_changed_words]
    for word, jaccard in results:
        print("word {word} has jaccard measure {jaccard}".format(word=word, jaccard=jaccard))
        print("word {word} has the following neighbors in model1:".format(word=word))
        print(*[word for word, score in w2v1.most_similar(word)], sep=",")
        print("word {word} has the following neighbors in model2:".format(word=word))
        print(*[word for word, score in w2v2.most_similar(word)], sep=",")
        print("==========================================================")


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
