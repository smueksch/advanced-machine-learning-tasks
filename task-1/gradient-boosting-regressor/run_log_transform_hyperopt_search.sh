module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=512]" \
-o log_transform_hyperopt_search.py \
"python log_transform_hyperopt_search.py --name 'GBR (Log. Transform Hyperopt Search)'"