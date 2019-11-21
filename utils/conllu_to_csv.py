import conllu
from tqdm import tqdm
import pandas as pd

for year in range(2015, 2020):
    corpus = open('corpora/conllu/{year}.conllu'.format(year=year))

    raw_texts = []
    lemmas = []

    parsed = conllu.parse_incr(corpus)
    for tokenlist in tqdm(parsed):
        tokens = [token['lemma'] + '_' + token['upostag'] for token in tokenlist]
        lemmas.append(' '.join(tokens))
        raw_texts.append(tokenlist.metadata.get('text'))

    df = pd.DataFrame({"LEMMAS": lemmas, "RAW": raw_texts})
    df.index.names = ['ID']
    df.to_csv('{year}_contexts.csv'.format(year=year))
