module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=1024]" \
-o grid_search.log \
"python grid_search.py --name 'XGBC (Grid Search)'"