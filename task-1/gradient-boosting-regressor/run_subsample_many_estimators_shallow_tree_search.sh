module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 64 \
-R "rusage[mem=512]" \
-o subsample_many_estimators_shallow_tree_search.log \
"python subsample_many_estimators_shallow_tree_search.py --name 'GBR (Many Estimators Shallow Tree Search)'"