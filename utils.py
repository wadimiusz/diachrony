import sys
import gensim


def informative_output(words_and_scores, w2v1: gensim.models.KeyedVectors, w2v2: gensim.models.KeyedVectors,
                       top_n_neighbors: int, model_name: str):
    print(model_name.center(40, '='))

    for word, score in words_and_scores:
        top_n_1 = [word for word, score in w2v1.most_similar(word, topn=top_n_neighbors)]
        top_n_2 = [word for word, score in w2v2.most_similar(word, topn=top_n_neighbors)]
        print("word {word} has score {score}".format(word=word, score=score))
        print("word {word} has the following neighbors in model1:".format(word=word))
        print(*top_n_1, sep=',')
        print('_' * 40)
        print("word {word} has the following neighbors in model2:".format(word=word))
        print(*top_n_2, sep=',')
        print("")


def simple_output(words_and_scores, model_name):
    print(model_name.center(40, '='))
    print(*[word for word, score in words_and_scores], sep='\n')
    print('')


def log(message: str, verbose: bool, end: str = '\n'):
    if verbose:
        sys.stderr.write(message+end)
        sys.stderr.flush()
