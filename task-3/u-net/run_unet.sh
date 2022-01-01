module load r/3.5.1 gcc/6.3.0 python_gpu/3.8.5 eth_proxy
bsub -W 4:00 \
-n 8 \
-R "rusage[mem=1024, ngpus_excl_p=1]" \
-o unet-log.txt \
"python unet.py --config unet.json"