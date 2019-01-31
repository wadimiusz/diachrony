python3 sampling.py --models wordvectors/*.model --pos-tag ADJ --top-n-most-frequent-words 1000 --shuffle --positive-samples 10 --negative-samples 10 | tee dataset/samples.txt 2> /dev/null
cat dataset/samples.txt | nice -n 20 python3 utils/creating_corpus.py | tee dataset/corpus.txt 2> /dev/null
