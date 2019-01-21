python3 sampling.py --models wordvectors/*.model --pos-tag NOUN --top-n-most-frequent-words 1000 --shuffle --positive-samples 10 --negative-samples 10 | tee samples.txt
cat samples.txt | nice -n 20 python3 creating_corpus.py | tee corpus.txt
