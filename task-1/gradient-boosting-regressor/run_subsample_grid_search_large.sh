module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=8192]" \
-o subsample_grid_search_large.log \
"python subsample_grid_search_large.py --name 'GBR (Large Subsample Grid Search)'"