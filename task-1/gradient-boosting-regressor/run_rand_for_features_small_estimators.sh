module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 24:00 \
-n 16 \
-R "rusage[mem=512]" \
-o rand_for_features_small_estimators.log \
"python rand_for_features_small_estimators.py --name 'GBR (Small Estimators Random Forest Features)'"