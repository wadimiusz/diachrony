**To get adjectives from corpus for comparing:**  

$ python3 get_adjs.py adjectives/rusentilex.csv ADJ 500 corpus_lengths_rus.tsv adjectives/rest/rusentilex_regular.csv adjectives/rest/rusentilex_incremental.csv adjectives/rest/rusentilex_regular_filtered_500.csv adjectives/rest/rusentilex_incremental_filtered_500.csv adjectives/rusentilex_regular_filtered_500.csv adjectives/rusentilex_incremental_filtered_500.csv

$ python3 get_adjs.py adjectives/ProductSentiRus.csv ADJ 500 corpus_lengths_rus.tsv adjectives/rest/sentirus_regular.csv adjectives/rest/sentirus_incremental.csv adjectives/rest/sentirus_regular_filtered_500.csv adjectives/rest/sentirus_incremental_filtered_500.csv adjectives/ProductSentiRus_regular_filtered_500.csv adjectives/ProductSentiRus_incremental_filtered_500.csv

**To evaluate adjectives:**  

$ python3 comparing.py adjectives/rusentilex_regular_filtered_500.csv adjectives/rusentilex_incremental_filtered_500.csv adjectives/rest/rusentilex_regular_filtered_500.csv adjectives/rest/rusentilex_incremental_filtered_500.csv outputs/rusentilex_eval_regular_500.csv outputs/rusentilex_eval_incremental_500.csv outputs/rusentilex_rest_regular_500.csv outputs/rusentilex_rest_incremental_500.csv corpus_lengths_rus.tsv

$ python3 comparing.py adjectives/ProductSentiRus_regular_filtered_500.csv adjectives/ProductSentiRus_incremental_filtered_500.csv adjectives/rest/sentirus_regular_filtered_500.csv adjectives/rest/sentirus_incremental_filtered_500.csv outputs/sentirus_eval_regular_500.csv outputs/sentirus_eval_incremental_500.csv outputs/sentirus_rest_regular_500.csv outputs/sentirus_rest_incremental_500.csv corpus_lengths_rus.tsv

**To plot results:**  

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv mean_dist_procrustes outputs/resuntilex_regular_mean_dist_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv mean_dist_globalanchors outputs/resuntilex_regular_mean_dist_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv mean_dist_procrustes outputs/resuntilex_incremental_mean_dist_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv mean_dist_globalanchors outputs/resuntilex_incremental_mean_dist_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv sum_deltas_procrustes outputs/resuntilex_regular_sum_deltas_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv sum_deltas_globalanchors outputs/resuntilex_regular_sum_deltas_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv sum_deltas_procrustes outputs/resuntilex_incremental_sum_deltas_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv sum_deltas_globalanchors outputs/resuntilex_incremental_sum_deltas_globalanchors.png

**T-test:**  

$ python3 ttest.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv

$ python3 ttest.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv
