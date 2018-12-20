alias python=python3.6
python --version
python naive_comparison.py --model1 models/2010.model --model2 models/2012.model --top-n-changed-words 50 --verbose --pos-tag ADJ --top-n-most-frequent-words 5000 --top-n-neighbors 50
