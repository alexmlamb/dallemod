
CUDA_VISIBLE_DEVICES=0 python3 main.py False gaussian >log_gaussian_nosn &

CUDA_VISIBLE_DEVICES=1 python3 main.py True gaussian >log_gaussian_sn &

CUDA_VISIBLE_DEVICES=2 python3 main.py False discrete >log_discrete_nosn &

CUDA_VISIBLE_DEVICES=3 python3 main.py True discrete >log_discrete_sn &



