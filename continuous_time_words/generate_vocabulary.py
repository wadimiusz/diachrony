from keras.preprocessing.sequence import skipgrams
import glob, tarfile
from collections import defaultdict
from smart_open import smart_open
import sys

#file_list = sorted(glob.glob("acl_time_data/*"))
vocab_list = []
min_freq = 25
max_vocab_size = 500000
vocabulary = defaultdict(float)
window_size = 5
negative_samples=2.0
min_year, max_year = 2010, 2017
max_doc_len = 5
max_sents = 1000000000

#for fname in file_list:
#    time = fname.split(".")[0]
#    time = (float(time)-2010)/2018

file_list = glob.glob("corpus*.txt.gz")

for fname in file_list:
    print("Processing file {}".format(fname))
    for i, line in enumerate(smart_open(fname, "r")):
        arr = line.strip().split(" ")

#        if len(arr) < max_doc_len: continue

        for word in arr:
            vocabulary[word] += 1
        if i%100000 == 0:
            print("Processed {} sentence".format(i), file=sys.stderr)

        if i > max_sents: break

for vocab, freq in vocabulary.items():
    if freq >= min_freq: vocab_list.append(vocab)

top_vocab_list = vocab_list[:max_vocab_size]

vocab_size = len(top_vocab_list)+2 #OOV_CHAR = 1, 0 is special character

with smart_open("vocabulary.txt.gz", "w") as f:
    print(*top_vocab_list, file=f, sep="\n")
