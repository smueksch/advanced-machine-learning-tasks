module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=512]" \
-o baseline_search.log \
"python baseline_search.py --name 'XGBR (Hyperparam. Search)'"