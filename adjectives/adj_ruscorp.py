from models import GlobalAnchors, ProcrustesAligner
from utils import load_model
import pandas as pd

results_anchors = pd.DataFrame()
results_procrustes = pd.DataFrame()

words = []
adjs = open('adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
for line in adjs.read().splitlines():
    words.append(line+'_ADJ')

results_anchors['word'] = words
results_procrustes['word'] = words

#Я не знаю как это сделать по-умному :(
years = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14']

def getscores(n1,n2):
    w2v1 = load_model('wordvectors/20{}.model'.format(n1))
    w2v2 = load_model('wordvectors/20{}.model'.format(n2))

    resultsGA = []
    resultsPr = []

    for word in words:
        try:
            global_anchors_result = GlobalAnchors(w2v1=w2v1, w2v2=w2v2).get_score(word)
            procrustes_result = ProcrustesAligner(w2v1=w2v1, w2v2=w2v2).get_score(word)
            resultsGA.append(global_anchors_result)
            resultsPr.append(procrustes_result)
        except KeyError:
            resultsGA.append(None)
            resultsPr.append(None)

    return resultsGA, resultsPr

for i in range(0,len(years)-1):
    results_anchors['{n1}-{n2}'.format(n1=years[i],n2=years[i+1])], results_procrustes['{n1}-{n2}'.format(n1=years[i],n2=years[i+1])] \
        = getscores(years[i],years[i+1])

results_anchors['mean'] = results_anchors[['00-01', '13-14']].mean(axis=1)
results_procrustes['mean'] = results_procrustes[['00-01', '13-14']].mean(axis=1)

#print(results_anchors.head(10))
#print(results_procrustes.head(10))

results_anchors.to_csv('globalanchors_result.csv', encoding='utf8')
results_procrustes.to_csv('procrustesaligner_result.csv', encoding='utf8')