#!/bin/sh
env="PCP"
algo="commformer_dec"
exp="single"
seed=1

echo "env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1"
CUDA_VISIBLE_DEVICES=$1 python train/train_pp.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --seed ${seed} --scenario_name ${env} --n_block 1 --n_embd 64 \
    --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --num_env_steps 10000000 \
    --ppo_epoch 10 --clip_param 0.05 --use_ReLU --gain 0.01 --lr 5e-4 --use_eval \
    --use_bilevel  --n_eval_rollout_threads 50 --eval_episodes 50 \
    --nfriendly_P 2 --nfriendly_A 1  --vision 1 
