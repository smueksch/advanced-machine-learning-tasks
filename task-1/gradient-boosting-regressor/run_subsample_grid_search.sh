module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=8192]" \
-o subsample_grid_search.log \
"python subsample_grid_search.py --name 'GBR (Subsample Grid Search)'"