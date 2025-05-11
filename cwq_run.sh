#!/usr/bin/env bash
#
# webqsp_reinforce_small_sample_freeze_then_train.sh
#
# Redirect all output up front

source ~/myenv/bin/activate

exec > preprocess_cwq.log 2>&1

echo "starting experiment on $(hostname) at $(date)"
echo "Shell: $SHELL; Python: $(which python3)"

# (any setup, module loads, env activates, etc.)
# source ~/.bashrc
# module load anaconda/2023a-pytorch
# source activate g_retriever

# --- wrap the training call in nohup & disown ---

nohup python3 -u -m src.dataset.preprocess.cwq > preprocess_cwq.out 2>&1 &

# nohup python3 -m src.dataset.cwq > postprocess_cwq.out 2>&1 &

# nohup python3 train_sl_retriever.py \
#     --dataset cwq \
#     --num_epochs 100 --max_steps 20 --learning_rate 3e-4 \
#     --directed False --triple_graph True \
#     --model_name cwq_first \
#     --sl_epochs 20 --sl_batch_size 32 --sl_minibatch_size 32 \
#     --hidden_dim 256 \
#     --use_pretrained_classifier \
#     > train_sl_retriever.out 2>&1 &

# nohup python3 train_retriever2.py \
#     --dataset cwq \
#     --num_epochs 100 --max_steps 30 --learning_rate 3e-4 \
#     --directed False --triple_graph True \
#     --model_name cwq_first \
#     --tf_start_bias 1 --tf_end_bias 1.0 --tf_total_epochs 1 \
#     --ppo_epochs 4 --ppo_batch_size 64 --ppo_minibatch_size 128 \
#     --sl_epochs 7 --sl_batch_size 32 --sl_minibatch_size 32 \
#     --hidden_dim 256 \
#     --use_pretrained_classifier \
#     --pretrained_classifier_path /home/user/G-Retriever/experiments/checkpoints/cwq_first/SL_classifier.pt \
#     --skip_pretrain \
#     > train_retriever2.out 2>&1 &


# nohup python3 train_retriever2.py \
#     --dataset webqsp \
#     --num_epochs 100 --max_steps 20 --learning_rate 3e-4 \
#     --directed False --triple_graph True \
#     --model_name GLASS_new_small_sample_freeze_then_train_mod_reward \
#     --tf_start_bias 1 --tf_end_bias 1.0 --tf_total_epochs 1 \
#     --ppo_epochs 4 --ppo_batch_size 64 --ppo_minibatch_size 128 \
#     --sl_epochs 7 --sl_batch_size 32 --sl_minibatch_size 32 \
#     --hidden_dim 256 \
#     --use_pretrained_classifier \
#     --pretrained_classifier_path /home/user/G-Retriever/experiments/checkpoints/GLASS_new_small_sample_freeze_then_train/SL_classifier.pt \
#     --skip_pretrain \
#     > train_retriever2.out 2>&1 &

# nohup python3 evaluate_retriever.py \
#     --dataset webqsp \
#     --model_name GLASS_new_small_sample_freeze_then_train_mod_reward \
#     --directed False --triple_graph True \
#     --max_steps 50 \
#     --hidden_dim 256 \
#     --use_pretrained_classifier \
#     --pretrained_classifier_path /home/user/G-Retriever/experiments/checkpoints/GLASS_new_small_sample_freeze_then_train/SL_classifier.pt \
#     > evaluate_retriever.out 2>&1 &

# detach it so it ignores SIGHUP
disown

echo "Launched preprocess_cwq.py (PID=$!)"
echo "Logs: preprocess_cwq.out"



#!/bin/bash

#exec > webqsp_reinforce_small_sample_freeze_then_train.log 2>&1

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o webqsp_reinforce_small_sample_freeze_then_train.log

#echo $SHELL
#source ~/.bashrc    

#module load anaconda/2023a-pytorch
#module load nccl

#source activate g_retriever

#nvidia-smi

# echo "starting experiment"
# which python3

#python -m src.dataset.preprocess.webqsp
#python -m src.dataset.webqsp
#WANDB_MODE=disabled python train.py --dataset webqsp --model_name graph_llm --llm_model_name 3b_chat
#WANDB_MODE=disabled python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 3b_chat

# WANDB_MODE=disabled python3 train_retriever2.py --dataset webqsp --num_epochs 1 --max_steps 50 --learning_rate 0.001 \
#                         --directed False --triple_graph True --model_name GLASS_new_small_sample_freeze_then_train --tf_start_bias 1 \
#                         --tf_end_bias 1.0 --tf_total_epochs 1 --ppo_epochs 10 --ppo_batch_size 16 \
#                         --ppo_minibatch_size 128 --num_workers 2 --sl_epochs 20 --sl_batch_size 32 \
#                         --sl_minibatch_size 32 --hidden_dim 256 #--use_pretrained_classifier


# WANDB_MODE=disabled python3 train_retriever2.py --dataset webqsp --num_epochs 1 --max_steps 50 --learning_rate 0.001 \
#                         --directed False --triple_graph True --model_name GLASS_new_small_sample_freeze_then_train --tf_start_bias 1 \
#                         --tf_end_bias 1.0 --tf_total_epochs 1 --ppo_epochs 10 --ppo_batch_size 16 \
#                         --ppo_minibatch_size 128 --num_workers 2 --sl_epochs 1 --sl_batch_size 32 \
#                         --sl_minibatch_size 32 --hidden_dim 256 --use_pretrained_classifier \
#                         --skip_pretrain --pretrained_classifier_path /home/gridsan/mhadjiivanov/meng/G-Retriever/experiments/checkpoints/GLASS_new/SL_classifier.pt


# WANDB_MODE=disabled python train_retriever2.py --dataset webqsp --num_epochs 1 --max_steps 100 --learning_rate 0.0001 \
#                         --directed False --triple_graph True --model_name GLASS_new_subgraphRAG_100_step --tf_start_bias 1 \
#                         --tf_end_bias 1.0 --tf_total_epochs 1 --ppo_epochs 10 --ppo_batch_size 16 \
#                         --ppo_minibatch_size 128 --num_workers 2 --sl_epochs 10 --sl_batch_size 32 \
#                         --sl_minibatch_size 32 --hidden_dim 256 --use_pretrained_classifier \
#                         --skip_pretrain --pretrained_classifier_path /home/gridsan/mhadjiivanov/meng/G-Retriever/experiments/checkpoints/GLASS_new/SL_classifier.pt

#python -m src.utils.question_rewriter
echo "done"