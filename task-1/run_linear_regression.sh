module load gcc/6.3.0 python_cpu/3.8.5 eth_proxy
bsub -W 4:00 \
-n 4 \
-R "rusage[mem=4096]" \
-o linear_regression.log \
"python linear_regression.py --name 'Linear Regression'"