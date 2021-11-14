module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 32 \
-R "rusage[mem=512]" \
-o outliers_removed_log_transform_grid_search.log \
"python outliers_removed_log_transform_grid_search.py --name 'XGBR (Outliers Removed, Log Transform Search)'"