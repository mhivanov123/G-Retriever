#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o webqsp_triple_version.log

echo $SHELL
source ~/.bashrc    

#module load anaconda/2023a-pytorch
#module load nccl

source activate g_retriever

#nvidia-smi

echo "starting experiment"
which python

#python -m src.dataset.preprocess.webqsp
#python -m src.dataset.webqsp
#WANDB_MODE=disabled python train.py --dataset webqsp --model_name graph_llm --llm_model_name 3b_chat
#WANDB_MODE=disabled python inference.py --dataset webqsp --model_name inference_llm --llm_model_name 3b_chat
#WANDB_MODE=disabled python train_retriever.py --dataset webqsp --max_steps 10 --curriculum 2 --curriculum_perc 1.0 --save_dir ./experiments/curriculum_2
#python -m src.utils.question_rewriter
#python -m analyze_model

WANDB_MODE=disabled python train_retriever.py \
    --num_epochs 100 \
    --tf_start_bias 10.0 \
    --tf_end_bias 1.0 \
    --tf_total_epochs 50 \
    --model_name "Retriever_linear_annealing" \
    --max_steps 10 \
    --save_dir ./experiments/triple_version \
    --triple_graph True \
    --directed False \
    --ppo_batch_size 10 \
    --ppo_epochs 10

#WANDB_MODE=disabled python train_retriever.py \
#    --num_epochs 20 \
#    --tf_start_bias 100.0 \
#    --tf_end_bias 1.0 \
#    --tf_total_epochs 10 \
#    --model_name "Retriever_linear_annealing" \
#    --max_steps 10 \
#    --save_dir ./experiments/teacher_linear_annealing_full_dataset
echo "done"
