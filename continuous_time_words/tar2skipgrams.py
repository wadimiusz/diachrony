from keras.preprocessing.sequence import skipgrams
import glob, tarfile, sys
from collections import defaultdict
from smart_open import smart_open

#file_list = sorted(glob.glob("acl_time_data/*"))
vocab_list = []
min_freq = 25
max_vocab_size = 500000
vocabulary = defaultdict(float)
window_size = 5
negative_samples=2.0
min_year, max_year = 2010, 2018
max_doc_len = 5
max_sents = 1000000000

file_list = glob.glob("corpus*.txt.gz")

top_vocab_list = smart_open("vocabulary.txt.gz", "r").read().split("\n")

vocab_size = len(top_vocab_list)+2 #OOV_CHAR = 1, 0 is special character

fw = smart_open("train_data.txt.gz", "w")

for fname in file_list:
    time = fname.split("_")[1]
    print("Processing: ", fname)
    time = (float(time)-min_year)/(max_year-min_year)

    pairs, labels = [], []

    for i, line in enumerate(smart_open(fname, "r")):
        arr = line.strip().split(" ")

#        if len(arr) < max_doc_len: continue

        seq = []

        for word in arr:
            if word not in top_vocab_list:
                seq.append(1)
            else:
                seq.append(top_vocab_list.index(word)+2)

        temp = skipgrams(seq, vocab_size, window_size=window_size, negative_samples=negative_samples)
        pairs += temp[0]
        labels += temp[1]
#        print(temp)

        if i%100000 == 0:
            for w_c, label in zip(pairs, labels):
                if len(w_c) > 0:
                    print(*w_c, time, label, sep="\t", file=fw)
            print("Processed {} sentence".format(i), file=sys.stderr)
            pairs, labels = [], []

            if i > max_sents: break


fw.close()


