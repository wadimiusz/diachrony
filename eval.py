import sys
import functools
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from models import GlobalAnchors
from models import ProcrustesAligner
from models import Jaccard
from models import KendallTau

from utils import intersection_align_gensim
from utils import load_model


@functools.lru_cache(maxsize=None)
def get_models_by_year(year: int, kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model1 = load_model('wordvectors/{year}.model'.format(year=year))
        model2 = load_model('wordvectors/{year}.model'.format(year=year+1))
    else:
        model1 = load_model('wordvectors/incremental/{year}_incremental.model'.format(year=year))
        model2 = load_model('wordvectors/incremental/{year}_incremental.model'.format(year=year+1))

    model1, model2 = intersection_align_gensim(model1, model2, pos_tag="ADJ", top_n_most_frequent_words=5000)
    return model1, model2


df = pd.read_csv('dataset/annotated.csv')
df_longterm = pd.read_csv('dataset/gold_kutuzov_kuzmenko_2017.tsv')

scores = dict()
for kind in ['regular', 'incremental']:
    for Scorer in [GlobalAnchors, ProcrustesAligner, KendallTau, Jaccard]:
        scores[(kind, Scorer)] = list()
        for idx, values in df.iterrows():
            print("{kind}, {scorer}, {idx} / {max}".format(kind=kind, scorer=str(Scorer), idx=idx, max=280),
                  file=sys.stderr)

            year = values["BASE_YEAR"]
            word = values["WORD"]

            model1, model2 = get_models_by_year(year, kind)
            score = Scorer(w2v1=model1, w2v2=model2, top_n_neighbors=50).get_score(word)
            scores[(kind, Scorer)].append(score)

        X = np.array(scores[(kind, Scorer)]).reshape((-1, 1))
        clf = DecisionTreeClassifier(max_depth=5).fit(X, y=df["GROUND_TRUTH"])
        print("Word vectors are {kind}, scores are from {scorer}".format(kind=kind, scorer=str(Scorer)))
        print("Sanity check")
        print(classification_report(df["GROUND_TRUTH"], clf.predict(X)))
