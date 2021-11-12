module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=512]" \
-o rand_for_features_hyperopt_search.log \
"python rand_for_features_hyperopt_search.py --name 'GBR (Random Forest Features)'"