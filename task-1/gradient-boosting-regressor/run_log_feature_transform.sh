module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 4:00 \
-n 16 \
-R "rusage[mem=512]" \
-o log_feature_transform.log \
"python log_feature_transform.py --name 'GBR (Log Feature Transform)'"