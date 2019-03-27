from models import GlobalAnchors, ProcrustesAligner
from utils import load_model, intersection_align_gensim
import pandas as pd

results_anchors = pd.DataFrame()
results_procrustes = pd.DataFrame()

words = []
adjs = open('adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
for line in adjs.read().splitlines():
    words.append(line+'_ADJ')

results_anchors['word'] = words
results_procrustes['word'] = words


model1 = load_model("wordvectors/soviet/pre-soviet.model")
model2 = load_model("wordvectors/soviet/soviet.model")
model22 = load_model("wordvectors/soviet/soviet.model")
model3 = load_model("wordvectors/soviet/post-soviet.model")

model1_aligned, model2_aligned = intersection_align_gensim(m1=model1, m2=model2)
model2_aligned2, model3_aligned = intersection_align_gensim(m1=model22, m2=model3)

resultsGA1 = []
resultsGA2 = []
resultsPr1 = []
resultsPr2 = []

for word in words:
    try:
        global_anchors_result1 = GlobalAnchors(w2v1=model1_aligned, w2v2=model2_aligned).get_score(word)
        global_anchors_result2 = GlobalAnchors(w2v1=model2_aligned2, w2v2=model3_aligned).get_score(word)
        procrustes_result1 = ProcrustesAligner(w2v1=model1_aligned, w2v2=model2_aligned).get_score(word)
        procrustes_result2 = ProcrustesAligner(w2v1=model2_aligned2, w2v2=model3_aligned).get_score(word)
        resultsGA1.append(global_anchors_result1)
        resultsPr1.append(procrustes_result1)
        resultsGA2.append(global_anchors_result2)
        resultsPr2.append(procrustes_result2)
    except KeyError:
        resultsGA1.append(None)
        resultsPr1.append(None)
        resultsGA2.append(None)
        resultsPr2.append(None)

results_anchors['pre-soviet'] = resultsGA1
results_anchors['soviet-post'] = resultsGA2
results_procrustes['pre-soviet'] = resultsPr1
results_procrustes['soviet-post'] = resultsPr2

results_anchors['mean'] = results_anchors[['pre-soviet', 'soviet-post']].mean(axis=1)
results_procrustes['mean'] = results_procrustes[['pre-soviet', 'soviet-post']].mean(axis=1)

#print(results_anchors.head(10))
#print(results_procrustes.head(10))

results_anchors.to_csv('globalanchors_result_soviet.csv', encoding='utf8')
results_procrustes.to_csv('procrustesaligner_result_soviet.csv', encoding='utf8')