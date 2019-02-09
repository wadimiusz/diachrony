import pandas as pd
import numpy as np
from utils import log, format_time
import functools
import random
import sys
import time


@functools.lru_cache()
def get_instances(word: str, year: (str, int, np.int)):
    def gen():
        with open('corpora/{year}_texts.txt'.format(year=year)) as f:
            for line in f:
                if word in line.strip().split():
                    yield line.strip().split()

    return list(gen())


def main():
    input_df = pd.read_csv(sys.stdin)
    old_contexts = list()
    new_contexts = list()
    start = time.time()
    for idx, word, year in input_df[['WORD', 'BASE_YEAR']].itertuples():
        if idx > 0:
            duration = time.time() - start
            ETA = format_time((duration / idx) * (len(input_df) - idx))
        else:
            ETA = "UNK"
        if '_' in word:
            word = word[:word.find('_')]
        log("{idx}/{total},{year},ETA {ETA},{word}".format(
            idx=idx, year=year, total=len(input_df), word=word.ljust(40),ETA=ETA
        ), end='\r')
        try:
            if len(get_instances(word, year)) > 5:
                five_old_samples = random.sample(get_instances(word, year), 5)
            else:
                five_old_samples = get_instances(word, year)

            if len(get_instances(word, year+1)) > 5:
                five_new_samples = random.sample(get_instances(word, year+1), 5)
            else:
                five_new_samples = get_instances(word, year+1)

            old_contexts.append(five_old_samples)
            new_contexts.append(five_new_samples)
        except ValueError:
            raise ValueError("Problem with", idx, word, year, "because not enough samples found")

    log("")
    log("This took", format_time(time.time() - start))
    output_df = pd.DataFrame({"ID": input_df["ID"], "WORD": input_df["WORD"], "BASE_YEAR": input_df["BASE_YEAR"],
                             "OLD_CONTEXTS": old_contexts, "NEW_CONTEXTS": new_contexts})
    output_df = output_df.set_index('ID')
    print(output_df.to_csv())


if __name__ == "__main__":
    main()
