import sys
import functools
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score

from models import GlobalAnchors
from models import ProcrustesAligner
from models import Jaccard
from models import KendallTau

from utils import load_model


def get_models_by_year(year: int, kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model1 = load_model('wordvectors/{year}.model'.format(year=year))
        model2 = load_model('wordvectors/{year}.model'.format(year=year+1))
    else:
        model1 = load_model('wordvectors/incremental/{year}_incremental.model'.format(year=year))
        model2 = load_model('wordvectors/incremental/{year}_incremental.model'.format(year=year+1))

    return model1, model2


def get_soviet_model(kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model1 = load_model("wordvectors/soviet/pre-soviet.model")
        model2 = load_model("wordvectors/soviet/soviet.model")
    else:
        model1 = load_model("wordvectors/soviet/pre-soviet_incremental.model")
        model2 = load_model("wordvectors/soviet/soviet_incremental.model")

    return model1, model2


df = pd.read_csv('dataset/annotated.csv')
df_longterm = pd.read_csv('dataset/gold_kutuzov_kuzmenko_2017.tsv', sep='\t', header=None)

f1_macro = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
f1_macro.index.name = "id"

f1_for_2 = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
f1_for_2.index.name = "id"

binary = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})
binary.index.name = "id"

# Choosing the classification algorithm
algo = LogisticRegression(class_weight='balanced', n_jobs=2, multi_class='multinomial',
                          solver='lbfgs')
# algo = DecisionTreeClassifier(max_depth=5)

for kind in ['regular', 'incremental']:
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
            for year in range(2000, 2014):
                model1, model2 = get_models_by_year(year, kind)
                scorer = Scorer(w2v1=deepcopy(model1), w2v2=deepcopy(model2), top_n_neighbors=50)
                for num, (idx, values) in enumerate(df[df["BASE_YEAR"] == year].iterrows()):
                    print("{kind}, {scorer}, {num} / {year}".format(kind=kind, scorer=str(Scorer), num=num, year=year),
                          file=sys.stderr)

                    word = values["WORD"]
                    score = scorer.get_score(word)
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

f1_macro.to_csv('outputs/f1_macro.csv')
f1_for_2.to_csv('outputs/f1_for_2.csv')
binary.to_csv('outputs/binary.csv')

algo = LogisticRegression(class_weight='balanced', n_jobs=2, multi_class='multinomial',
                          solver='lbfgs')

soviet_output = pd.DataFrame(
    {"model": ["GlobalAnchors", "ProcrustesAligner", "KendallTau", "Jaccard", "united"]})

for kind in ["regular", "incremental"]:
    model1, model2 = get_soviet_model(kind)
    scorers = [GlobalAnchors, ProcrustesAligner, KendallTau, Jaccard]
    X = np.ndarray((len(df_longterm), len(scorers)))
    y_true = df_longterm[1]
    current_f1_macro = list()

    current_f1_for_2 = list()
    scores = list()
    for scorer_num in range(len(scorers) + 1):
        if scorer_num != len(scorers):
            Scorer = scorers[scorer_num]
            scorer = Scorer(w2v1=deepcopy(model1), w2v2=deepcopy(model2), top_n_neighbors=50)
            for idx, word in enumerate(df_longterm[0]):
                if scorer_num != len(scoregrs):
                    X[idx, scorer_num] = scorer.get_score(word)

        fold_creator = StratifiedKFold(9, shuffle=False)

        current_scores = list()
        for train_idx, test_idx in fold_creator.split(X, y_true):
            if scorer_num != len(scorers):
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

            if len(np.unique(y_train)) != 2:
                raise ValueError(np.unique(y_train))

            if len(np.unique(y_test)) != 2:
                raise ValueError(np.unique(y_test))

            if len(np.unique(y_pred)) != 2:
                raise ValueError(np.unique(y_pred))
            current_scores.append(f1_score(y_test, y_pred, average='macro'))

        scores.append(np.mean(current_scores))

    soviet_output[kind] = scores

soviet_output.to_csv("outputs/soviet_f1_macro.csv")