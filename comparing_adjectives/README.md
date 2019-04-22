**To get adjectives from corpus for comparing:**  

$ python3 get_adjs.py adjectives/rusentilex.csv ADJ 500 corpus_lengths_rus.tsv adjectives/rest/rusentilex/ (with_distribution | simple)

$ python3 get_adjs.py adjectives/ProductSentiRus.csv ADJ 500 corpus_lengths_rus.tsv adjectives/rest/sentirus/ (with_distribution | simple)

**To evaluate adjectives:**  

$ python3 comparing_adjectives/comparing.py -l rusentilex -k regular -mf 500 -lg comparing_adjectives/corpus_lengths_rus.tsv -n 50

**To plot results:**  

$ python3 plot_results.py outputs/rusentilex/eval_regular_500.csv outputs/rusentilex/rest_regular_500.csv (mean_dist_procrustes outputs/rusentilex/regular_mean_dist_procrustes_500.png | mean_dist_globalanchors outputs/rusentilex/regular_mean_dist_globalanchors_500.png)

$ python3 plot_results.py outputs/rusentilex/eval_incremental_500.csv outputs/rusentilex/rest_incremental_500.csv (mean_dist_procrustes outputs/rusentilex/incremental_mean_dist_procrustes_500.png | mean_dist_globalanchors outputs/rusentilex/incremental_mean_dist_globalanchors_500.png)

$ python3 plot_results.py outputs/rusentilex/eval_regular_500.csv outputs/rusentilex/rest_regular_500.csv (sum_deltas_procrustes outputs/rusentilex/regular_deltas_procrustes_500.png | sum_deltas_globalanchors outputs/rusentilex/regular_deltas_globalanchors_500.png)

$ python3 plot_results.py outputs/rusentilex/eval_incremental_500.csv outputs/rusentilex/rest_incremental_500.csv (sum_deltas_procrustes outputs/rusentilex/incremental_deltas_procrustes_500.png | sum_deltas_globalanchors outputs/rusentilex/incremental_deltas_globalanchors_500.png)

**T-test:**  

$ python3 ttest.py outputs/rusentilex/eval_regular_500.csv outputs/rusentilex/rest_regular_500.csv

$ python3 ttest.py outputs/rusentilex/eval_incremental_500.csv outputs/rusentilex/rest_incremental_500.csv

$ python3 ttest.py outputs/sentirus/eval_regular_500.csv outputs/sentirus/rest_regular_500.csv

$ python3 ttest.py outputs/sentirus/eval_incremental_500.csv outputs/sentirus/rest_incremental_500.csv
