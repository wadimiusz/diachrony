from keras.preprocessing.sequence import skipgrams
import glob
from collections import defaultdict

file_list = sorted(glob.glob("acl_time_data/*"))
vocab_list = []
min_freq = 5
max_vocab_size = 500000
vocabulary = defaultdict(float)
window_size = 5
negative_samples=2.0
min_year, max_year = 2010, 2018

for fname in file_list:
#    time = fname.split(".")[0]
#    time = (float(time)-2010)/2018
    for line in open(fname, "r"):
        arr = line.strip().split(" ")
        for word in arr:
            vocabulary[word] += 1

for vocab, freq in vocabulary.items():
    if freq >= min_freq: vocab_list.append(vocab)

top_vocab_list = vocab_list[:max_vocab_size]

vocab_size = len(top_vocab_list)+2 #OOV_CHAR = 1, 0 is special character

fw = open("train_data.txt", "w")

for fname in file_list:
    time = fname.split(".")[0].split("/")[-1]
    print("Processing: ", fname)
    time = (float(time)-min_year)/(max_year-min_year)
    for line in open(fname, "r"):
        arr = line.strip().split(" ")
        if len(arr) < 3: continue
        seq = []
        for word in arr:
            if word not in top_vocab_list:
                seq.append(1)
            else:
                seq.append(top_vocab_list.index(word)+2)
        sg = skipgrams(seq, vocab_size, window_size=window_size, negative_samples=negative_samples)

        for w_c, label in zip(sg[0], sg[1]):
            if len(w_c) > 0:
                print(*w_c, time, label, sep="\t", file=fw)
fw.close()

with open("vocabulary.txt", "w") as f:
    print(*top_vocab_list, file=f, sep="\n")
