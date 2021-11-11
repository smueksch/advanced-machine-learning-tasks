module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 8:00 \
-n 16 \
-R "rusage[mem=512]" \
-o auto_contamination.log \
"python auto_contamination.py --name 'LOF (Auto Contamination Search)'"