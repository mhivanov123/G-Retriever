#!/bin/bash

# Slurm sbatch options
#SBATCH -n 20
#SBATCH --gres=gpu:volta:1
#SBATCH -o webqsp_rewrite_questions_GLASS_8_teacher_forcing_no_grad_clip.log

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
WANDB_MODE=disabled python train_retriever.py --dataset webqsp
#python -m src.utils.question_rewriter
echo "done"
