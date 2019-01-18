python3 sampling.py --models wordvectors/*.model --pos-tag ADJ --top-n-most-frequent-words 1000 --shuffle --positive-samples 10 --negative-samples 10 2> /dev/null | tee corpus.txt
