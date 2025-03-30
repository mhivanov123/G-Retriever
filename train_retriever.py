import os
import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate

from src.models.retrieval_policy import RetrievalPolicy
from src.models.rl_retriever import RetrievalTrainer

from src.dataset.webqsp import WebQSPDataset
import logging
from pathlib import Path
from src.utils.collate import collate_fn

from transformers import AutoModel, AutoTokenizer

HF_DIR = "hf"
HF_MODELS_DIR = os.path.join(HF_DIR, "models")
HF_DATASETS_DIR = os.path.join(HF_DIR, "datasets")
os.makedirs(HF_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_DIR, exist_ok=True)

#pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
pretrained_repo = os.path.join(HF_MODELS_DIR, 'all-roberta-large-v1')

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_dir = Path(save_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Training configuration
    config = {
        "node_dim": 1024,  # assuming BERT embeddings
        "hidden_dim": 256,
        "num_heads": 4,
        "batch_size": 32,
        "num_epochs": 100,
        "max_steps": 100,
        "episodes_per_state": 1
    }
    
    # Setup logging
    logger = setup_logging("./experiments")
    logger.info(f"Training config: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize dataset
    dataset = WebQSPDataset() #load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    train_indices = idx_split['train']
    val_indices = idx_split['val']
    
    # Initialize model and trainer
    policy_net = RetrievalPolicy(
        node_dim=config["node_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"]
    ).to(device)

    model = AutoModel.from_pretrained(pretrained_repo).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    def encoder(text):
        # Tokenize the input text
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        
        # Move tokens to same device as model
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get embeddings from model
        with torch.no_grad():
            embeddings = model(input_ids=tokens['input_ids'], 
                             attention_mask=tokens['attention_mask']).last_hidden_state.mean(dim=1)
        return embeddings

    trainer = RetrievalTrainer(policy_net, encoder=encoder)
    
    # Training loop with validation
    num_training_steps = config["num_epochs"] * len(train_indices) * config["episodes_per_state"]
    progress_bar = tqdm(range(num_training_steps))
    best_val_reward = float('-inf')
    
    for epoch in range(config["num_epochs"]):
        # Training
        train_stats = []
        policy_net.train()

        if epoch > config["num_epochs"]*0.5:
            trainer.set_teacher_forcing(False)
        else:
            trainer.set_teacher_forcing(True)

        
        for idx in train_indices:

            for i in range(config["episodes_per_state"]):


                sample = dataset[idx]
                # Move batch to device
                graph = sample['graph'].to(device)
                q_nodes = sample['q_idx'] if 'q_idx' in sample else None
                a_nodes = sample['a_idx'] if 'a_idx' in sample else None
                
                if q_nodes is None or len(q_nodes) == 0 or a_nodes is None or len(a_nodes) == 0:
                    logger.warning("No question or answer nodes in this batch")
                    continue

                if graph.x.float().isnan().sum().item() > 0:
                    logger.warning("NaN values in graph")
                    continue
                
                # Run training step
                stats = trainer.train_step(
                    sample,
                    max_steps=config["max_steps"]
                )
                
                if stats is not None:
                    train_stats.append(stats)

                progress_bar.update(1)
           
        # Compute average training stats
        if train_stats:
            avg_train_stats = {
                k: sum(s[k] for s in train_stats) / len(train_stats)
                for k in train_stats[0].keys()
            }
            logger.info(f"Epoch {epoch} - Training:")
            logger.info(f"  Loss: {avg_train_stats['loss']:.4f}")
            logger.info(f"  Reward: {avg_train_stats['reward']:.4f}")
            logger.info(f"  Steps: {avg_train_stats['steps']:.2f}")
            #logger.info(f"  Correct Nodes: {avg_train_stats['correct_nodes']:.2f}")
            logger.info(f"  Perc Correct Nodes: {avg_train_stats['perc_correct_nodes']:.2f}")
            logger.info(f"  Perc Answer Nodes Reached: {avg_train_stats['perc_answer_nodes_reached']:.2f}")
            logger.info(f"  Visited Nodes: {avg_train_stats['visited_nodes']:.2f}")
            logger.info(f"  Visited Edges: {avg_train_stats['visited_edges']:.2f}")
        else:
            logger.warning("No successful training steps in this epoch")
            #continue
        
        # Validation
        val_stats = []
        policy_net.eval()
        
        with torch.no_grad():
            for idx in val_indices:
                sample = dataset[idx]
                # Move batch to device
                graph = sample['graph'].to(device)
                q_nodes = sample['q_idx'] if 'q_entity' in sample else None
                a_nodes = sample['a_idx'] if 'a_entity' in sample else None
                shortest_path_nodes = sample['shortest_path_nodes'] if 'shortest_path_nodes' in sample else None
                
                if q_nodes is None or len(q_nodes) == 0 or a_nodes is None or len(a_nodes) == 0:
                    continue
                
                # Get subgraph through inference
                visited_nodes, visited_edges = trainer.inference_step(
                    sample=sample,
                    max_steps=config["max_steps"]
                )

                nodes_in_subgraph = sum(1 for node in shortest_path_nodes if node in visited_nodes)
                nodes_not_in_subgraph = sum(1 for node in shortest_path_nodes if node not in visited_nodes)
                
                nodes_in_subgraph_percentage = nodes_in_subgraph / len(visited_nodes)
                nodes_not_in_subgraph_percentage = nodes_not_in_subgraph / len(visited_nodes)

                # Check if any target nodes were reached
                success = any(node in visited_nodes for node in a_nodes)
                if success:
                    val_stats.append({
                        'success': 1.0,
                        'nodes': len(visited_nodes),
                        'edges': len(visited_edges)
                    })
                else:
                    val_stats.append({'success': 0.0, 'nodes': len(visited_nodes), 'edges': len(visited_edges)})
        
        # Compute validation metrics
        if val_stats:
            avg_val_stats = {
                k: sum(s[k] for s in val_stats) / len(val_stats)
                for k in val_stats[0].keys()
            }
            logger.info(f"Epoch {epoch} - Validation:")
            logger.info(f"  Success Rate: {avg_val_stats['success']:.4f}")
            logger.info(f"  Avg Nodes: {avg_val_stats['nodes']:.2f}")
            logger.info(f"  Avg Edges: {avg_val_stats['edges']:.2f}")
            
            # Save best model based on success rate
            if avg_val_stats['success'] > best_val_reward:
                best_val_reward = avg_val_stats['success']
                save_path = Path("./experiments/checkpoints")
                save_path.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy_net.state_dict(),
                    'val_success': best_val_reward,
                    'config': config
                }, save_path / "best_model.pt")
                
                logger.info(f"Saved new best model with validation success rate: {best_val_reward:.4f}")
        else:
            logger.warning("No successful validation steps in this epoch")

if __name__ == "__main__":
    main()