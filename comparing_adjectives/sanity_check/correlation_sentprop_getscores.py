from scipy.stats import pearsonr
import pandas as pd
from models import GlobalAnchors, smart_procrustes_align_gensim, Jaccard
import numpy as np
import logging
from utils import intersection_align_gensim
import os
import gensim

sentprop_regular = pd.read_csv('sentprop_regular.csv')
sentprop_incremental = pd.read_csv('sentprop_incremental.csv')

words_regular = sentprop_regular['WORD'].tolist()
words_incremental = sentprop_incremental['WORD'].tolist()


def load_model(embeddings_file):
    """
    This function, unifies various standards of word embedding files.
    It automatically determines the format by the file extension
    and loads it from the disk correspondingly.
    :param embeddings_file: path to the file
    :return: the loaded model
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if not os.path.isfile(embeddings_file):
        raise FileNotFoundError("No file called {file}".format(file=embeddings_file))
    # Determine the model format by the file extension
    # Binary word2vec file:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    # Text word2vec file:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)

    return emb_model


def get_models_by_decade(cur_decade: int, kind: str):
    if kind == "regular":
        model = load_model('../data_eng/{decade}.model'.format(decade=cur_decade))
    else:
        model = load_model('../data_eng/{decade}_incremental.model'.format(decade=cur_decade))

    return model


models_regular = []
models_incremental = []

for decade in range(1960, 2010, 10):
    model_regular = get_models_by_decade(decade, 'regular')
    model_incremental = get_models_by_decade(decade, 'incremental')

    models_regular.append(model_regular)
    models_incremental.append(model_incremental)

vocabs_incremental = [model.vocab for model in models_incremental]
intersec_vocabs_incremental = set.intersection(*map(set, vocabs_incremental))

vocabs_regular = [model.vocab for model in models_regular]
intersec_vocabs_regular = set.intersection(*map(set, vocabs_regular))


def intersec_models(modellist, intersec_vocab):
    for model in modellist[1:]:
        _, _ = intersection_align_gensim(m1=modellist[0], m2=model, words=intersec_vocab)

    return modellist


def align_models(modellist):
    for model in modellist[1:]:
        _ = smart_procrustes_align_gensim(modellist[0], model)

    return modellist


def get_jaccard(wordlist, model1, model2, top_n_neighbors):
    scores = {}
    for word in wordlist:
        score = Jaccard(w2v1=model1, w2v2=model2,
                            top_n_neighbors=top_n_neighbors).get_score(word)
        scores[word] = (1 - score)

    return scores


def get_globalanchors(wordlist, model1, model2):
    scores = {}
    for word in wordlist:
        score = GlobalAnchors(w2v1=model1, w2v2=model2, assume_vocabs_are_identical=True).get_score(word)
        scores[word] = (1 - score)

    return scores


def get_procrustes(wordlist, model1, model2):
    scores = {}
    for word in wordlist:
        score = np.dot(model1[word], model2[word])
        scores[word] = (1 - score)

    return scores


jaccard_incremental = pd.DataFrame({'WORD': words_incremental})
jaccard_incremental['60-70'] = jaccard_incremental['WORD'].map(get_jaccard(words_incremental, models_incremental[0], models_incremental[1], 10))
jaccard_incremental['70-80'] = jaccard_incremental['WORD'].map(get_jaccard(words_incremental, models_incremental[1], models_incremental[2], 10))
jaccard_incremental['80-90'] = jaccard_incremental['WORD'].map(get_jaccard(words_incremental, models_incremental[2], models_incremental[3], 10))
jaccard_incremental['90-00'] = jaccard_incremental['WORD'].map(get_jaccard(words_incremental, models_incremental[3], models_incremental[4], 10))
jaccard_incremental.to_csv('jaccard_incremental.csv')

jaccard_regular = pd.DataFrame({'WORD': words_regular})
jaccard_regular['60-70'] = jaccard_regular['WORD'].map(get_jaccard(words_regular, models_regular[0], models_regular[1], 10))
jaccard_regular['70-80'] = jaccard_regular['WORD'].map(get_jaccard(words_regular, models_regular[1], models_regular[2], 10))
jaccard_regular['80-90'] = jaccard_regular['WORD'].map(get_jaccard(words_regular, models_regular[2], models_regular[3], 10))
jaccard_regular['90-00'] = jaccard_regular['WORD'].map(get_jaccard(words_regular, models_regular[3], models_regular[4], 10))
jaccard_regular.to_csv('jaccard_regular.csv')

intersected_models_regular = intersec_models(models_regular, intersec_vocabs_regular)
intersected_models_incremental = intersec_models(models_incremental, intersec_vocabs_incremental)


ga_incremental = pd.DataFrame({'WORD': words_incremental})
ga_incremental['60-70'] = ga_incremental['WORD'].map(get_globalanchors(words_incremental, intersected_models_incremental[0], intersected_models_incremental[1]))
ga_incremental['70-80'] = ga_incremental['WORD'].map(get_globalanchors(words_incremental, intersected_models_incremental[1], intersected_models_incremental[2]))
ga_incremental['80-90'] = ga_incremental['WORD'].map(get_globalanchors(words_incremental, intersected_models_incremental[2], intersected_models_incremental[3]))
ga_incremental['90-00'] = ga_incremental['WORD'].map(get_globalanchors(words_incremental, intersected_models_incremental[3], intersected_models_incremental[4]))
ga_incremental.to_csv('globalanchors_incremental.csv')

ga_regular = pd.DataFrame({'WORD': words_regular})
ga_regular['60-70'] = ga_regular['WORD'].map(get_globalanchors(words_regular, intersected_models_regular[0], intersected_models_regular[1]))
ga_regular['70-80'] = ga_regular['WORD'].map(get_globalanchors(words_regular, intersected_models_regular[1], intersected_models_regular[2]))
ga_regular['80-90'] = ga_regular['WORD'].map(get_globalanchors(words_regular, intersected_models_regular[2], intersected_models_regular[3]))
ga_regular['90-00'] = ga_regular['WORD'].map(get_globalanchors(words_regular, intersected_models_regular[3], intersected_models_regular[4]))
ga_regular.to_csv('globalanchors_regular.csv')


cos_incremental = pd.DataFrame({'WORD': words_incremental})
cos_incremental['60-70'] = cos_incremental['WORD'].map(get_procrustes(words_incremental, intersected_models_incremental[0], intersected_models_incremental[1]))
cos_incremental['70-80'] = cos_incremental['WORD'].map(get_procrustes(words_incremental, intersected_models_incremental[1], intersected_models_incremental[2]))
cos_incremental['80-90'] = cos_incremental['WORD'].map(get_procrustes(words_incremental, intersected_models_incremental[2], intersected_models_incremental[3]))
cos_incremental['90-00'] = cos_incremental['WORD'].map(get_procrustes(words_incremental, intersected_models_incremental[3], intersected_models_incremental[4]))
cos_incremental.to_csv('cosine_incremental.csv')

aligned_models = align_models(intersected_models_regular)

procrustes = pd.DataFrame({'WORD': words_regular})
procrustes['60-70'] = procrustes['WORD'].map(get_procrustes(words_regular, aligned_models[0], aligned_models[1]))
procrustes['70-80'] = procrustes['WORD'].map(get_procrustes(words_regular, aligned_models[1], aligned_models[2]))
procrustes['80-90'] = procrustes['WORD'].map(get_procrustes(words_regular, aligned_models[2], aligned_models[3]))
procrustes['90-00'] = procrustes['WORD'].map(get_procrustes(words_regular, aligned_models[3], aligned_models[4]))
procrustes.to_csv('procrustes_regular.csv')