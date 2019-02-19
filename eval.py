import sys
import functools
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

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
f1_macro = pd.DataFrame({"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
f1_macro.index.name="id"

f1_for_2 = pd.DataFrame({"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
f1_for_2.index.name="id"
for kind in ['regular', 'incremental']:
    max_scorers = 4
    max_samples = len(df)
    X = np.ndarray((max_samples, max_scorers))
    y = np.array(df["GROUND_TRUTH"])
    current_f1_macro = list()
    current_f1_for_2 = list()
    for scorer_num, Scorer in enumerate([GlobalAnchors, ProcrustesAligner, KendallTau, Jaccard]):
        for idx, values in df.iterrows():
            print("{kind}, {scorer}, {idx} / {max}".format(kind=kind, scorer=str(Scorer), idx=idx, max=280),
                  file=sys.stderr)

            year = values["BASE_YEAR"]
            word = values["WORD"]

            model1, model2 = get_models_by_year(year, kind)
            score = Scorer(w2v1=model1, w2v2=model2, top_n_neighbors=50).get_score(word)
            X[idx, scorer_num] = score

        clf = DecisionTreeClassifier(max_depth=5)
        fold = StratifiedKFold(9, shuffle=False)
        model_f1 = np.mean(cross_val_score(clf, X[:, scorer_num].reshape(-1, 1), y, scoring='f1_macro', cv=fold))
        current_f1_macro.append(model_f1)
        model_f1_for_2 = np.mean(cross_val_score(
            clf, X[:, scorer_num].reshape(-1, 1), y, scoring=lambda estimator, X, y: f1_score(
                estimator.predict(X), y, average='macro', pos_label=1, labels=[1]), cv=fold))

        current_f1_for_2.append(model_f1_for_2)

    clf = DecisionTreeClassifier(max_depth=5)
    fold = StratifiedKFold(9, shuffle=False)
    united_f1 = np.mean(cross_val_score(clf, X, y, scoring='f1_macro', cv=fold))
    current_f1_macro.append(united_f1)
    f1_macro[kind] = current_f1_macro
    united_f1_for_2 = np.mean(cross_val_score(
            clf, X, y, scoring=lambda estimator, X, y: f1_score(
                estimator.predict(X), y, average='macro', pos_label=1, labels=[1]), cv=fold))

    current_f1_for_2.append(united_f1_for_2)
    f1_for_2[kind] = current_f1_for_2

f1_macro.to_csv('outputs/f1_macro.csv')
f1_for_2.to_csv('outputs/f1_for_2.csv')
