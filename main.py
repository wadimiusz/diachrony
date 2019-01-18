import argparse
import gensim
import random
import numpy as np

from models import get_changes_by_jaccard
from models import get_changes_by_kendalltau
from models import get_changes_by_procrustes
from models import get_changes_by_global_anchors
from utils import log
from utils import informative_output, simple_output
from utils import load_model
from utils import intersection_align_gensim


def word_frequency(model: gensim.models.KeyedVectors, word: str) -> int:
    """
    A handy function for extracting the word frequency from models
    :param model: the model in question
    :param word: extract the frequency of this word
    :return:
    """
    return model.wv.vocab[word].count


def comparison(w2v1_path: str, w2v2_path: str, top_n_neighbors: int,
               top_n_changed_words: (int, None), top_n_most_frequent_words: (int, None), pos_tag: (str, None),
               informative: bool):
    """
    This module extracts two models from two specified paths and compares the meanings of words within their vocabulary.
    :param w2v1_path: the path to the first model
    :param w2v2_path: the path to the second modet
    :param top_n_neighbors: we will compare top n neighbors of words
    :param top_n_changed_words: we will output top n most interesting words, may be int or None
    :param top_n_most_frequent_words: we will use top n most frequent words from each model, may be int or None
    :param pos_tag: specify this to consider only words with a specific pos_tag
    :param informative: if True we use informative_output for printing output (more verbose and interpretable)
    :return: None
    """
    w2v1 = load_model(w2v1_path)
    w2v2 = load_model(w2v2_path)

    log("The first model contains {words1} words, e. g. {word1}\n"
        "The second model contains {words2} words, e. g. {word2}".format(
               words1=len(w2v1.wv.vocab), words2=len(w2v2.wv.vocab), word1=random.choice(list(w2v1.wv.vocab.keys())),
               word2=random.choice(list(w2v2.wv.vocab.keys()))))

    w2v1, w2v2 = intersection_align_gensim(w2v1, w2v2, pos_tag=pos_tag,
                                           top_n_most_frequent_words=top_n_most_frequent_words)

    log("After preprocessing, the first model contains {words1} words, e. g. {word1}\n"
        "The second model contains {words2} words, e. g. {word2}".format(
               words1=len(w2v1.wv.vocab), words2=len(w2v2.wv.vocab), word1=random.choice(list(w2v1.wv.vocab.keys())),
               word2=random.choice(list(w2v2.wv.vocab.keys()))))

    jaccard_result = get_changes_by_jaccard(w2v1=w2v1, w2v2=w2v2, top_n_changed_words=top_n_changed_words,
                                            top_n_neighbors=top_n_neighbors)

    kendalltau_result = get_changes_by_kendalltau(w2v1=w2v1, w2v2=w2v2, top_n_changed_words=top_n_changed_words,
                                                  top_n_neighbors=top_n_neighbors)

    procrustes_result = get_changes_by_procrustes(w2v1=w2v1, w2v2=w2v2, top_n_changed_words=top_n_changed_words)

    global_anchors_result = get_changes_by_global_anchors(w2v1=w2v1, w2v2=w2v2, top_n_changed_words=top_n_changed_words)

    results = (jaccard_result, kendalltau_result, procrustes_result, global_anchors_result)
    names = ('JACCARD', 'KENDALL TAU', 'PROCRUSTES', 'GLOBAL ANCHORS')

    if informative:
        for result, name in zip(results, names):
            informative_output(result, w2v1, w2v2, top_n_neighbors, name)
    else:
        for result, name in zip(results, names):
            simple_output(result, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, help="the path to the first model", required=True)
    parser.add_argument('--model2', type=str, help="the path to the second model", required=True)
    parser.add_argument("--top-n-neighbors", type=int, default=10, help="We will compare top n nearest neighbors",
                        dest="top_n_neighbors")

    parser.add_argument("--top-n-changed-words", type=int, default=None, help="We will output top n changed words",
                        dest="top_n_changed_words")

    parser.add_argument("--top-n-most-frequent-words", type=int, default=None,
                        help="We will output top n changed words", dest="top_n_most_frequent_words")

    parser.add_argument("--pos-tag", type=str, default=None, help="If you want to see words with a specific "
                                                                  "pos tag, specify this", dest="pos_tag")

    parser.add_argument('--informative-output', action='store_true', dest='informative_output',
                        help='This argument makes the output more verbose and interpretable')
 
    args = parser.parse_args()
    comparison(w2v1_path=args.model1, w2v2_path=args.model2, top_n_neighbors=args.top_n_neighbors,
               top_n_most_frequent_words=args.top_n_most_frequent_words, pos_tag=args.pos_tag,
               top_n_changed_words=args.top_n_changed_words, informative=args.informative_output)


if __name__ == "__main__":
    main()
