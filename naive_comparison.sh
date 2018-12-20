alias python=python3.6
python --version
python naive_comparison.py --model1 models/2013.model --model2 models/2014.model --top-n-changed-words 10 --verbose --pos-tag ADJ --top-n-most-frequent-words 1000 --top-n-neighbors 50
