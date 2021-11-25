module load gcc/6.3.0 python/3.8.5 eth_proxy
bsub -W 4:00 \
-n 8 \
-R "rusage[mem=1024]" \
-o gbc.log \
"python gbc.py --name 'GBC (Basic AutoEncoder)'"