import os
import wandb
from tqdm import tqdm
import torch
import argparse
from src.utils.seed import seed_everything
import random
from src.models.retrieval_policy import RetrievalPolicy, RetrievalPolicyTriple, RetrievalPolicyTriple2, SL_warmup_policy
from src.models.rl_retriever import RetrievalTrainer
import torch.nn.functional as F
from src.dataset.webqsp import WebQSPDataset
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.nn as nn
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a retrieval policy network on graph data")
    
    # Model parameters
    parser.add_argument("--node_dim", type=int, default=1024, help="Node embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for GNN layers")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--directed", type=bool, default=False, help="Whether to use directed edges")
    parser.add_argument("--triple_graph", type=bool, default=False, help="Whether to use triple graph")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps in each episode")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    
    # Teacher forcing parameters
    parser.add_argument("--tf_start_bias", type=float, default=10.0, help="Initial teacher forcing bias")
    parser.add_argument("--tf_end_bias", type=float, default=1.0, help="Final teacher forcing bias")
    parser.add_argument("--tf_total_epochs", type=int, default=50, help="Total epochs over which to anneal teacher forcing")
    parser.add_argument("--tf_schedule", type=str, default="linear", choices=["linear", "exp", "cosine"], help="Annealing schedule type")
    
    # PPO parameters
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs")
    parser.add_argument("--ppo_batch_size", type=int, default=8, help="Number of PPO batch size")
    parser.add_argument("--ppo_minibatch_size", type=int, default=8, help="Number of PPO minibatch size")

    # SL parameters
    parser.add_argument("--sl_epochs", type=int, default=5, help="Number of SL epochs")
    parser.add_argument("--sl_batch_size", type=int, default=8, help="Number of SL batch size")
    parser.add_argument("--sl_minibatch_size", type=int, default=8, help="Number of SL minibatch size")

    # Parallel parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    
    # Curriculum parameters
    parser.add_argument("--curriculum", type=float, default=float('inf'), help="Curriculum for the model")
    parser.add_argument("--curriculum_perc", type=float, default=0.2, help="Percentage of epochs to use curriculum")

    # Paths and logging
    parser.add_argument("--save_dir", type=str, default="./experiments", help="Directory to save logs and checkpoints")
    parser.add_argument("--model_name", type=str, default="GLASS_GCN", help="Name for the model")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="graph-retriever", help="Wandb project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset to use")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use (cuda/cpu)")
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Setup logging
    logger = setup_logging(args.save_dir)
    logger.info(f"Training config: {args}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Initialize dataset
    dataset = WebQSPDataset(directed=args.directed, triple=args.triple_graph)
    idx_split = dataset.get_idx_split()

    train_indices = idx_split['train']
    val_indices = idx_split['val']
    
    node_dim = args.node_dim
    if args.triple_graph:
        node_dim = node_dim

    # Initialize model and trainer
    if args.triple_graph:
        policy_net = RetrievalPolicyTriple2(
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        ).to(device)
    else:
        policy_net = RetrievalPolicy(
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads
        ).to(device)
    
    trainer = RetrievalTrainer(
        policy_net,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size,
        ppo_minibatch_size=args.ppo_minibatch_size,
        tf_start_bias=args.tf_start_bias,
        tf_end_bias=args.tf_end_bias, 
        tf_total_epochs=args.tf_total_epochs,
        directed=args.directed,
        triple_graph=args.triple_graph,
        num_workers=args.num_workers
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.save_dir) / "checkpoints" / f"{args.model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop with validation
    num_training_steps = args.num_epochs * len(train_indices)
    progress_bar = tqdm(range(num_training_steps))
    best_val_reward = float('-inf')

    # Pretrain
    logger.info("Pretraining Start")

    sl_warmup_policy = SL_warmup_policy(policy_net, hidden_dim=args.hidden_dim).to(device)
    sl_optimizer = torch.optim.Adam(sl_warmup_policy.parameters(), lr=1e-4)

    pretrain_indices = idx_split['train']
    val_indices = idx_split['val']

    for epoch in range(args.sl_epochs):
        epoch_loss = 0

        for idx in pretrain_indices:

            sample = dataset[idx]
            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue
            trajectory = trainer.build_pretrain_trajectory(sample, policy_net, device, args.max_steps)
            if trajectory is None:
                logger.warning(f"Trajectory {idx} is None")
                continue
            
            sl_optimizer.zero_grad()

            y_pred = sl_warmup_policy(x_ = trajectory['x'].float(), 
                                        edge_index = trajectory['edge_index'].long(), 
                                        edge_attr = trajectory['edge_attr'].float(), 
                                        question_embeddings = trajectory['question_embeddings'], 
                                        subgraph_mask = trajectory['subgraph_mask'], 
                                        action_mask = trajectory['action_mask'])
            
            y = trajectory['y']

            #weight = torch.ones_like(y)
            #weight[y == 1] = y.shape[0] / y.sum().item()
            #logger.info(f"y_pred: {y_pred.shape}, y: {y.shape}")

            #loss = F.binary_cross_entropy(y_pred.squeeze(-1), 
            #                                y.float(),
            #                                weight=weight)
            pos_w = torch.tensor(y.shape[0] / y.sum().item(), device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='sum')  # w0 is implicitly 1
            loss = criterion(y_pred.squeeze(-1), y.float()) / y.shape[0]

            loss.backward()
            sl_optimizer.step()
            epoch_loss += loss.item()
            
        logger.info(f"Epoch {epoch} loss: {epoch_loss}")

        sl_warmup_policy.eval()
        
        tp, fp, tn, fn = 0, 0, 0, 0
        total_samples = 0
        for idx in val_indices:
            sample = dataset[idx]
            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue
            trajectory = trainer.build_pretrain_trajectory(sample, policy_net, device, args.max_steps)
            if trajectory is None:
                logger.warning(f"Trajectory {idx} is None")
                continue
            y_pred = sl_warmup_policy(x_ = trajectory['x'].float(), 
                                        edge_index = trajectory['edge_index'].long(), 
                                        edge_attr = trajectory['edge_attr'].float(), 
                                        question_embeddings = trajectory['question_embeddings'], 
                                        subgraph_mask = trajectory['subgraph_mask'], 
                                        action_mask = trajectory['action_mask'])
            
            y = trajectory['y']

            weight = torch.ones_like(y)
            weight[y == 1] = 100.0

            #loss = F.binary_cross_entropy(y_pred[trajectory['action_mask'].nonzero()].squeeze(-1), 
            #                            y[trajectory['action_mask'].nonzero()].float(),
            #                            weight=weight[trajectory['action_mask'].nonzero()])

            predictions = (y_pred > 0.5).float().squeeze(-1)
            tp += ((predictions == 1) & (y == 1)).sum().item()
            fp += ((predictions == 1) & (y == 0)).sum().item()
            tn += ((predictions == 0) & (y == 0)).sum().item()
            fn += ((predictions == 0) & (y == 1)).sum().item()
            total_samples += y.shape[0]

        precision = tp / max(tp + fp, 1e-6)
        recall = tp / max(tp + fn, 1e-6)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
        logger.info(f"Epoch {epoch} - Validation:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"True positives: {tp}, False positives: {fp}, True negatives: {tn}, False negatives: {fn}")

        sl_warmup_policy.train()

    logger.info("Pretraining End")

    for epoch in range(args.num_epochs):
        # Update current epoch in trainer for annealing calculation
        trainer.set_current_epoch(epoch)
        
        # Log the current teacher bias
        current_bias = trainer.get_current_teacher_bias()
        logger.info(f"Epoch {epoch} - Current teacher bias: {current_bias:.4f}")

        # Training
        train_stats = []
        policy_net.train()
        
        for idx in train_indices:
            sample = dataset[idx]

            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue

            #curriculum enforcement
            if epoch <= args.num_epochs * args.curriculum_perc and 'shortest_path_nodes' in sample and len(sample['shortest_path_nodes']) > args.curriculum:
                continue

            # Move sample to device
            graph = sample['graph'].to(device)
            q_idx = sample['q_idx'] if 'q_idx' in sample else None
            a_idx = sample['a_idx'] if 'a_idx' in sample else None
            shortest_path_nodes = sample['shortest_path_nodes'] if 'shortest_path_nodes' in sample else None

            if not q_idx or not a_idx or not shortest_path_nodes:
                logger.warning("No question or answer nodes in this sample")
                continue

            if graph.x.float().isnan().sum().item() > 0:
                logger.warning("NaN values in graph")
                continue
            
            # Run training step
            stats = trainer.train_step(
                sample,
                max_steps=args.max_steps
            )

            if stats is not None:
                train_stats.append(stats)

            progress_bar.update(1)

            if len(train_stats)% 100 == 0:
                logger.info(f"Epoch {epoch}, {len(train_stats)-100} - {len(train_stats)}, Train stats")
                avg_train_stats = {
                k: sum(s[k] for s in train_stats[-100:]) / len(train_stats[-100:])
                for k in train_stats[-100:][0].keys()
                }
                logger.info(f"  Loss: {avg_train_stats['loss']:.4f}")
                logger.info(f"  Policy Loss: {avg_train_stats['policy_loss']:.4f}")
                logger.info(f"  Value Loss: {avg_train_stats['value_loss']:.4f}")
                logger.info(f"  Entropy: {avg_train_stats['entropy']:.4f}")
                logger.info(f"  Reward: {avg_train_stats['reward']:.4f}")
                logger.info(f"  Steps: {avg_train_stats['steps']:.2f}")
                #logger.info(f"  Correct Nodes: {avg_train_stats['correct_nodes']:.2f}")
                logger.info(f"  Perc Correct Nodes: {avg_train_stats['perc_correct_nodes']:.2f}")
                logger.info(f"  Perc Answer Nodes Reached: {avg_train_stats['perc_answer_nodes_reached']:.2f}")
                logger.info(f"  Perc Any Answer Node Reached: {avg_train_stats['perc_any_answer_node_reached']:.2f}")
                logger.info(f"  Visited Nodes: {avg_train_stats['visited_nodes']:.2f}")
                logger.info(f"  Visited Edges: {avg_train_stats['visited_edges']:.2f}")

        logger.info(f"Epoch {epoch} stats")
        
        # Compute average training stats
        if train_stats:
            avg_train_stats = {
                k: sum(s[k] for s in train_stats) / len(train_stats)
                for k in train_stats[0].keys()
            }
            logger.info(f"Epoch {epoch} - Training:")
            logger.info(f"  Loss: {avg_train_stats['loss']:.4f}")
            logger.info(f"  Policy Loss: {avg_train_stats['policy_loss']:.4f}")
            logger.info(f"  Value Loss: {avg_train_stats['value_loss']:.4f}")
            logger.info(f"  Entropy: {avg_train_stats['entropy']:.4f}")
            logger.info(f"  Reward: {avg_train_stats['reward']:.4f}")
            logger.info(f"  Steps: {avg_train_stats['steps']:.2f}")
            #logger.info(f"  Correct Nodes: {avg_train_stats['correct_nodes']:.2f}")
            logger.info(f"  Perc Correct Nodes: {avg_train_stats['perc_correct_nodes']:.2f}")
            logger.info(f"  Perc Answer Nodes Reached: {avg_train_stats['perc_answer_nodes_reached']:.2f}")
            logger.info(f"  Perc Any Answer Node Reached: {avg_train_stats['perc_any_answer_node_reached']:.2f}")
            logger.info(f"  Visited Nodes: {avg_train_stats['visited_nodes']:.2f}")
            logger.info(f"  Visited Edges: {avg_train_stats['visited_edges']:.2f}")
            
            if args.use_wandb:
                wandb.log({f"train/{k}": v for k, v in avg_train_stats.items()})
        else:
            logger.warning("No successful training steps in this epoch")
        
        # Validation
        val_stats = []
        policy_net.eval()
        
        with torch.no_grad():
            for idx in val_indices:
                sample = dataset[idx]

                if sample is None:
                    logger.warning(f"Sample {idx} is None")
                    continue
                # Move sample to device
                graph = sample['graph'].to(device)
                q_nodes = sample['q_idx'] if 'q_idx' in sample else None
                a_nodes = sample['a_idx'] if 'a_idx' in sample else None 
                shortest_path_nodes = sample['shortest_path_nodes'] if 'shortest_path_nodes' in sample else None
                
                if not args.triple_graph and (q_nodes is None or len(q_nodes) == 0 or a_nodes is None or len(a_nodes) == 0):
                    logger.warning("No question or answer nodes in this validation sample")
                    continue
                
                # Get subgraph through inference
                visited_nodes, visited_edges = trainer.inference_step(
                    sample=sample,
                    max_steps=args.max_steps
                )

                # Calculate success and stats
                nodes_in_subgraph = sum(1 for node in shortest_path_nodes if node in visited_nodes) if shortest_path_nodes else 0
                nodes_not_in_subgraph = sum(1 for node in shortest_path_nodes if node not in visited_nodes) if shortest_path_nodes else 0
                
                if shortest_path_nodes:
                    nodes_in_subgraph_percentage = nodes_in_subgraph / len(visited_nodes) if visited_nodes else 0
                    nodes_not_in_subgraph_percentage = nodes_not_in_subgraph / len(visited_nodes) if visited_nodes else 0
                else:
                    nodes_in_subgraph_percentage = 0
                    nodes_not_in_subgraph_percentage = 0

                # Check if any target nodes were reached
                success = any(node in visited_nodes for node in a_nodes)
                
                val_stats.append({
                    'success': 1.0 if success else 0.0,
                    'nodes': len(visited_nodes),
                    'edges': len(visited_edges)
                })
        
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
            
            if args.use_wandb:
                wandb.log({f"val/{k}": v for k, v in avg_val_stats.items()})
            
            # Save model checkpoint if it's the best so far
            if avg_val_stats['success'] > best_val_reward:
                best_val_reward = avg_val_stats['success']
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy_net.state_dict(),
                    'val_success': best_val_reward,
                    'config': vars(args)
                }, checkpoint_dir / "best_model.pt")
                
                logger.info(f"Saved new best model with validation success rate: {best_val_reward:.4f}")
            
            # Save regular checkpoint every save_every epochs
            if epoch % args.save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy_net.state_dict(),
                    'val_success': avg_val_stats['success'],
                    'config': vars(args)
                }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
                
                logger.info(f"Saved checkpoint at epoch {epoch}")
        else:
            logger.warning("No successful validation steps in this epoch")
    
    # Final logging
    logger.info(f"Training completed. Best validation success rate: {best_val_reward:.4f}")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    main()