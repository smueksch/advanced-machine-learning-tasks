module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 8:00 \
-n 16 \
-R "rusage[mem=512]" \
-o float_contamination.log \
"python float_contamination.py --name 'IsolFor (Float Contamination Search)'"