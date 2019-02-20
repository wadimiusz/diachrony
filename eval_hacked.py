import sys
import functools
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from copy import deepcopy
from models import GlobalAnchors
from models import ProcrustesAligner
from models import Jaccard
from models import KendallTau
from utils import load_model


@functools.lru_cache(maxsize=None)
def get_model(name):
    return load_model(name)


def get_models_by_year(year: int, kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model1 = get_model('wordvectors/{year}.model'.format(year=year))
        model2 = get_model('wordvectors/{year}.model'.format(year=year + 1))
    else:
        model1 = get_model('wordvectors/incremental/{year}_incremental.model'.format(year=year))
        model2 = get_model('wordvectors/incremental/{year}_incremental.model'.format(year=year + 1))

    return model1, model2


def get_soviet_model(kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model1 = get_model("wordvectors/soviet/pre-soviet.model")
        model2 = get_model("wordvectors/soviet/soviet.model")
    else:
        model1 = get_model("wordvectors/soviet/pre-soviet_incremental.model")
        model2 = get_model("wordvectors/soviet/soviet_incremental.model")

    return model1, model2


modeltype = sys.argv[1]

df = pd.read_csv('dataset/annotated.csv')
df_longterm = pd.read_csv('dataset/gold_kutuzov_kuzmenko_2017.tsv')

f1_macro = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
f1_macro.index.name = "id"

f1_for_2 = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
f1_for_2.index.name = "id"

binary = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
binary.index.name = "id"

algo = LogisticRegression(class_weight='balanced', n_jobs=2, multi_class='multinomial',
                          solver='lbfgs')
# algo = DummyClassifier()

for kind in [modeltype]:
    max_scorers = 4
    max_samples = len(df)
    X = np.ndarray((max_samples, max_scorers))
    y_true = np.array(df["GROUND_TRUTH"])
    current_f1_macro = list()
    current_f1_for_2 = list()
    scores = {"f1_macro": list(), "f1_for_2": list(), "binary": list()}
    scorers = [GlobalAnchors, ProcrustesAligner, KendallTau, Jaccard]
    for scorer_num in [0, 1, 2, 3, None]:
        if scorer_num is not None:
            Scorer = scorers[scorer_num]
            for idx, values in df.iterrows():
                print(
                    "{kind}, {scorer}, {idx} / {max}".format(kind=kind, scorer=str(Scorer), idx=idx,
                                                             max=280),
                    file=sys.stderr)

                year = values["BASE_YEAR"]
                word = values["WORD"]

                model1, model2 = get_models_by_year(year, kind)
                if scorer_num == 0 or scorer_num == 1:
                    big_lumpy_object = Scorer(w2v1=deepcopy(model1), w2v2=deepcopy(model2),
                                              top_n_neighbors=50)
                    score = big_lumpy_object.get_score(word)
                    del big_lumpy_object
                else:
                    score = Scorer(w2v1=model1, w2v2=model2, top_n_neighbors=50).get_score(word)
                X[idx, scorer_num] = score

        fold_creator = StratifiedKFold(9, shuffle=False)
        current_scores = {"f1_macro": list(), "f1_for_2": list(), "binary": list()}
        for train_idx, test_idx in fold_creator.split(X[:, scorer_num], y_true):
            if scorer_num is not None:
                X_train = X[:, scorer_num][train_idx]
                X_test = X[:, scorer_num][test_idx]
                X_train = X_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)
            else:
                X_train = X[train_idx]
                X_test = X[test_idx]
                print(X_train.shape, X_test.shape)

            y_train = y_true[train_idx]
            y_test = y_true[test_idx]
            clf = algo.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            unique, counts = np.unique(y_true, return_counts=True)
            if min(counts) == 0:
                raise ValueError
            print("True", np.asarray((unique, counts)).T)
            unique, counts = np.unique(y_pred, return_counts=True)
            print("Predicted", np.asarray((unique, counts)).T)
            if min(counts) == 0:
                raise ValueError
            current_scores["f1_macro"].append(f1_score(y_test, y_pred, average='macro'))
            current_scores["f1_for_2"].append(f1_score(y_test, y_pred, labels=[2], average='macro'))

            y_train_binary = (y_train > 0).astype(int)
            y_test_binary = (y_test > 0).astype(int)
            binary_clf = algo.fit(X_train, y_train_binary)
            y_pred_binary = binary_clf.predict(X_test)
            current_scores["binary"].append(f1_score(y_test_binary, y_pred_binary))

        scores["f1_macro"].append(np.mean(current_scores["f1_macro"]))
        scores["f1_for_2"].append(np.mean(current_scores["f1_for_2"]))
        scores["binary"].append(np.mean(current_scores["binary"]))
    f1_macro[kind] = scores["f1_macro"]
    f1_for_2[kind] = scores["f1_for_2"]
    binary[kind] = scores["binary"]

print('================')
print('F1')
print(f1_macro)
print('================')
print('================')
print('F1 for class 2')
print(f1_for_2)
print('================')
print('================')
print('F1 binary')
print(binary)
print('================')
