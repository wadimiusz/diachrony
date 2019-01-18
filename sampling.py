import argparse
import random
import pandas as pd

from utils import load_model
from utils import intersection_align_gensim
from models import get_changes_by_global_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='paths to models to compare pairwise')
    parser.add_argument('--pos-tag', default=None, type=str, help='specify this to remove words with other pos tags',
                        dest='pos_tag')
    parser.add_argument('--top-n-most-frequent-words', default=None, type=int,
                        help='you can specify n so that both positive and negative samples are from '
                             'top n most frequent words', dest="top_n_most_frequent_words")
    parser.add_argument('--positive-samples', type=int, default=10, help='words that have changed most',
                        dest='positive_samples')
    parser.add_argument('--negative-samples', type=int, default=10, help='randomly samples words',
                        dest='negative_samples')
    parser.add_argument('--shuffle', action='store_true', help='use this argument to shuffle the output')
    args = parser.parse_args()

    samples = list()
    labels = list()
    base_model_names = list()
    ratings = list()

    for model1_path, model2_path in zip(args.models, args.models[1:]):
        model1 = load_model(model1_path)
        model2 = load_model(model2_path)
        model1, model2 = intersection_align_gensim(model1, model2, pos_tag=args.pos_tag,
                                                   top_n_most_frequent_words=args.top_n_most_frequent_words)
        global_anchors_result = get_changes_by_global_anchors(model1, model1, args.positive_samples, verbose=False)
        positive_samples = [word for (word, score) in global_anchors_result]
        possible_negative_samples = set(model1.wv.vocab.keys()) - set(positive_samples)
        negative_samples = random.sample(possible_negative_samples, args.negative_samples)

        samples.extend(positive_samples)
        samples.extend(negative_samples)

        labels.extend([1] * args.positive_samples)
        labels.extend([0] * args.negative_samples)

        ratings.extend(range(1, args.positive_samples + 1))
        ratings.extend([-1] * args.negative_samples)

        base_model_names.extend([model1_path] * (args.positive_samples + args.negative_samples))

    output = pd.DataFrame({"WORD": samples, "LABEL": labels, 'ASSESSOR_LABEL': -1, 'LEFT_MODEL': base_model_names,
                           "RATING": ratings})
    output.index.names = ['ID']

    if args.shuffle:
        output = output.sample(frac=1).reset_index(drop=True)

    print(output.to_csv())


if __name__ == "__main__":
    main()
