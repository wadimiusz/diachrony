import argparse
import random
import pandas as pd
from scipy.stats import percentileofscore

from utils import load_model
from utils import intersection_align_gensim
from utils import log
from models import get_changes_by_global_anchors


def get_decile(word, vocab):
    percentile = percentileofscore([vocab[word].count for word in vocab], vocab[word].count)
    return int(percentile / 10)


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
    base_year = list()
    ratings = list()

    for num, (model1_path, model2_path) in enumerate(zip(args.models, args.models[1:])):
        log("{num} / {total} {model1} {model2}".format(num=num, total=len(args.models)-1, model1=model1_path,
                                                       model2=model2_path), end='\r')

        log('Done')
        model1 = load_model(model1_path)
        model2 = load_model(model2_path)
        model1, model2 = intersection_align_gensim(model1, model2, pos_tag=args.pos_tag,
                                                   top_n_most_frequent_words=args.top_n_most_frequent_words)
        global_anchors_result = get_changes_by_global_anchors(model1, model1, args.positive_samples)
        positive_samples = [word for (word, score) in global_anchors_result]
        possible_negative_samples = set(model1.wv.vocab.keys()) - set(positive_samples)
        negative_samples = list()
        for positive_sample in positive_samples:
            number_of_bin = get_decile(positive_sample, model1.vocab)
            eligible_negative_samples = [word for word in possible_negative_samples
                                         if get_decile(word, model1.vocab) == number_of_bin]

            negative_sample = random.choice(eligible_negative_samples)
            negative_samples.append(negative_sample)

        negative_samples = random.sample(possible_negative_samples, args.negative_samples)

        samples.extend(positive_samples)
        samples.extend(negative_samples)

        labels.extend([1] * args.positive_samples)
        labels.extend([0] * args.negative_samples)

        ratings.extend(range(1, args.positive_samples + 1))
        ratings.extend([-1] * args.negative_samples)

        if model1_path.startswith('wordvectors/') and model1_path.endswith('.model'):
            year = int(model1_path[len('wordvectors/'):-len('.model')])
        else:
            raise ValueError("Pattern of {path} is not recognized. Path to model must start with 'wordvectors/'"
                             " and end with '.model'. Feel free to change the pattern in the source code.".format(
                path=model1_path
            ))

        base_year.extend([year] * (args.positive_samples + args.negative_samples))

    output = pd.DataFrame({"WORD": samples, "LABEL": labels, 'ASSESSOR_LABEL': -1, 'BASE_YEAR': base_year,
                           "RATING": ratings})

    if args.shuffle:
        output = output.sample(frac=1).reset_index(drop=True)  # this shuffles the dataframe but not its index

    output.index.names = ['ID']
    print(output.to_csv())


if __name__ == "__main__":
    main()
