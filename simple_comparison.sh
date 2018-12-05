alias python=python3.6
python --version
python naive_comparison.py --model1 models/ruwikiruscorpora_0_300_20.bin --binary1 --model2 models/ruwikiruscorpora-nobigrams_upos_skipgram_300_5_2018.vec --top-n-changed-words 10 --top-n-most-frequent-words 1000 --pos-tag ADJ --verbose
