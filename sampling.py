import argparse
import random

from utils import load_model
from utils import intersection_align_gensim
from models import get_changes_by_global_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, help='path to the model1 for sampling')
    parser.add_argument('--model2', type=str, help='path to the model2 for sampling')
    parser.add_argument('--pos-tag', default=None, type=str, help='specify this to remove words with other pos tags',
                        dest='pos_tag')
    parser.add_argument('--top-n-most-frequent-words', default=None, type=int,
                        help='you can specify n so that both positive and negative samples are from '
                             'top n most frequent words', dest="top_n_most_frequent_words")
    parser.add_argument('--positive-samples', type=int, default=25, help='words that have changed most',
                        dest='positive_samples')
    parser.add_argument('--negative-samples', type=int, default=25, help='randomly samples words',
                        dest='negative_samples')
    parser.add_argument('--shuffle', action='store_true', help='use this argument to shuffle the output')
    args = parser.parse_args()

    model1 = load_model(args.model1)
    model2 = load_model(args.model2)
    model1, model2 = intersection_align_gensim(model1, model2, pos_tag=args.pos_tag,
                                               top_n_most_frequent_words=args.top_n_most_frequent_words)
    global_anchors_result = get_changes_by_global_anchors(model1, model1, args.positive_samples, verbose=False)
    positive_samples = [word for (word, score) in global_anchors_result]
    possible_negative_samples = set(model1.wv.vocab.keys()) - set(positive_samples)
    negative_samples = random.sample(possible_negative_samples, args.negative_samples)

    result = [(sample, 1) for sample in positive_samples]
    result.extend([(sample, 0) for sample in negative_samples])

    if args.shuffle:
        random.shuffle(result)

    for sample, label in result:
        print('{sample},{label}'.format(sample=sample, label=label))

if __name__ == "__main__":
    main()
