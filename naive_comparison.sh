alias python=python3.6
python --version
python naive_comparison.py --model1 models/news_win20.model.bin --binary1 --model2 models/news_0_300_2.bin --binary2 --top-n-changed-words 10 --verbose --cut-tags
