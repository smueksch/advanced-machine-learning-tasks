module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 32 \
-R "rusage[mem=512]" \
-o outlier_pre_selection_search.log \
"python outlier_pre_selection_search.py --name 'LOF (Pre-Select Features Search)'"