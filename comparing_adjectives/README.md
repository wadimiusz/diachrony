**To get adjectives from corpus for comparing:**  

python3 get_adjs.py -list_of_evaluative_adjectives -pos_tag -treshold_frequency -corpus_lengths | -output_files

$ python3 get_adjs.py adjectives/rusentilex.csv ADJ 500 10006254 9966851 9115530 19812333 39442865 adjectives/rest/rusentilex_regular.csv adjectives/rest/rusentilex_incremental.csv adjectives/rest/rusentilex_regular_filtered.csv adjectives/rest/rusentilex_incremental_filtered.csv adjectives/rusentilex_regular_filtered.csv adjectives/rusentilex_incremental_filtered.csv

$ python3 get_adjs.py adjectives/ProductSentiRus.csv ADJ 500 10006254 9966851 9115530 19812333 39442865 adjectives/rest/sentirus_regular.csv adjectives/rest/sentirus_incremental.csv adjectives/rest/sentirus_regular_filtered.csv adjectives/rest/sentirus_incremental_filtered.csv adjectives/ProductSentiRus_regular_filtered.csv adjectives/ProductSentiRus_incremental_filtered.csv

**To evaluate adjectives:**  

$ python3 comparing.py adjectives/rusentilex_regular_filtered.csv adjectives/rusentilex_incremental_filtered.csv adjectives/rest/rusentilex_regular_filtered.csv adjectives/rest/rusentilex_incremental_filtered.csv outputs/rusentilex_eval_regular.csv outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_regular.csv outputs/rusentilex_rest_incremental.csv 10006254 9966851 9115530 19812333 39442865

$ python3 comparing.py adjectives/ProductSentiRus_regular_filtered.csv adjectives/ProductSentiRus_incremental_filtered.csv adjectives/rest/sentirus_regular_filtered.csv adjectives/rest/sentirus_incremental_filtered.csv outputs/sentirus_eval_regular.csv outputs/sentirus_eval_incremental.csv outputs/sentirus_rest_regular.csv outputs/sentirus_rest_incremental.csv 10006254 9966851 9115530 19812333 39442865

**To plot results:**  

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv mean_dist_procrustes outputs/resuntilex_regular_mean_dist_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv mean_dist_globalanchors outputs/resuntilex_regular_mean_dist_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv mean_dist_procrustes outputs/resuntilex_incremental_mean_dist_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv mean_dist_globalanchors outputs/resuntilex_incremental_mean_dist_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv sum_deltas_procrustes outputs/resuntilex_regular_sum_deltas_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv sum_deltas_globalanchors outputs/resuntilex_regular_sum_deltas_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv sum_deltas_procrustes outputs/resuntilex_incremental_sum_deltas_procrustes.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv sum_deltas_globalanchors outputs/resuntilex_incremental_sum_deltas_globalanchors.png

$ python3 plot_results.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv std_from_meanvec outputs/resuntilex_regular_std.png

$ python3 plot_results.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv std_from_meanvec outputs/resuntilex_incremental_std.png

**T-test:**  

$ python3 ttest.py outputs/rusentilex_eval_regular.csv outputs/rusentilex_rest_regular.csv

$ python3 ttest.py outputs/rusentilex_eval_incremental.csv outputs/rusentilex_rest_incremental.csv
