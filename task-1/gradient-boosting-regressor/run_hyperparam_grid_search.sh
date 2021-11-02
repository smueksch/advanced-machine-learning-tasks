module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 8:00 \
-n 8 \
-R "rusage[mem=4096]" \
-o hyperparameter_grid_search.log \
"python hyperparam_grid_search.py --name 'GBR (Hyperparam. Grid Search)'"