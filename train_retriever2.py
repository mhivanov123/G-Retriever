import os
import wandb
from tqdm import tqdm
import torch
import argparse
from src.utils.seed import seed_everything
import random
from src.models.retrieval_policy import RetrievalPolicy, RetrievalPolicyTriple, RetrievalPolicyTriple2, SL_warmup_policy, SL_classifier, Policy, Policy_test, Policy_freeze_then_train
from src.models.rl_retriever import RetrievalTrainer, build_pretrain_trajectory
import torch.nn.functional as F
from src.dataset.webqsp import WebQSPDataset, CWQDataset
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import torch.nn as nn
import torch.optim as optim

torch.cuda.empty_cache()

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


    # Pretrained classifier
    parser.add_argument("--use_pretrained_classifier", action="store_true", help="Whether to use pretrained classifier")
    parser.add_argument("--skip_pretrain", action="store_true", help="Whether to skip pretraining")
    parser.add_argument("--pretrained_classifier_path", type=str, default=None, help="Path to pretrained classifier")

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
    if args.dataset == 'webqsp':
        dataset = WebQSPDataset(directed=args.directed, triple=args.triple_graph)
    elif args.dataset == 'cwq':
        dataset = CWQDataset(directed=args.directed, triple=args.triple_graph)
    idx_split = dataset.get_idx_split()

    train_indices = idx_split['train']
    #train_indices = []
    val_indices = idx_split['val']
    
    node_dim = args.node_dim
    if args.triple_graph:
        node_dim = node_dim

    # Create checkpoint directory
    checkpoint_dir = Path(args.save_dir) / "checkpoints" / f"{args.model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop with validation
    num_training_steps = args.num_epochs * len(train_indices)
    progress_bar = tqdm(range(num_training_steps))
    best_val_reward = float('-inf')

    # Pretrain
    logger.info("Pretraining Start")

    sl_classifier = SL_classifier(node_dim=args.node_dim, hidden_dim=args.hidden_dim).to(device)
    #test_sl_classifier = SL_classifier(node_dim=args.node_dim, hidden_dim=args.hidden_dim).to(device)
    #sl_optimizer = torch.optim.Adam(sl_classifier.parameters(), lr=1e-4)

    if args.pretrained_classifier_path is not None:
        sl_classifier.load_state_dict(torch.load(args.pretrained_classifier_path))
        #test_sl_classifier.load_state_dict(torch.load(args.pretrained_classifier_path))
    else:
        sl_classifier.load_state_dict(torch.load(checkpoint_dir / "SL_classifier.pt"))
            #test_sl_classifier.load_state_dict(torch.load(checkpoint_dir / "SL_classifier.pt"))

    if args.skip_pretrain:
        logger.info("Skipping pretraining")
    else:
        pretrain_classifier(sl_classifier, sl_optimizer, dataset, train_indices, val_indices, device, logger, args, checkpoint_dir)

    # Freeze the SL classifier  
    '''for param in sl_classifier.parameters():
        param.requires_grad = False'''

    '''for param in sl_classifier.parameters():
        param.requires_grad = True'''

    logger.info("Pretraining End")

    #policy_net = Policy_test(pretrained_classifier=sl_classifier, node_dim=args.node_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    #policy_net = Policy(pretrained_classifier=sl_classifier, node_dim=args.node_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    policy_net = Policy_freeze_then_train(pretrained_classifier=sl_classifier, node_dim=args.node_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    #policy_net_test = Policy_test(pretrained_classifier=test_sl_classifier, node_dim=args.node_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    
    for param in policy_net.parameters():
        param.requires_grad = True
    
    for param in policy_net.pretrained_classifier.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_net.parameters()), lr=args.learning_rate)

    #train_value_head_test(policy_net, optimizer, dataset, train_indices, val_indices, device, logger, args, checkpoint_dir)
    #train_value_head(policy_net, policy_net_test, optimizer, dataset, train_indices, val_indices, device, logger, args, checkpoint_dir)
    
    trainer = RetrievalTrainer(
        policy_net,
        optimizer,
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

    logger.info("Policy Training Start")
    # Training loop
    for epoch in range(args.num_epochs):
        # Shuffle training indices at the beginning of each epoch
        # This helps prevent overfitting and improves generalization
        random.shuffle(train_indices)
        logger.info(f"Shuffled {len(train_indices)} training indices for epoch {epoch}")
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
                for k in train_stats[-100:][0].keys() if k != 'entropy'
                }
                avg_entropy = sum([sum(s['entropy']) for s in train_stats[-100:]]) / sum([len(s['entropy']) for s in train_stats[-100:]])

                logger.info(f"  Loss: {avg_train_stats['loss']:.4f}")
                logger.info(f"  Policy Loss: {avg_train_stats['policy_loss']:.4f}")
                logger.info(f"  Value Loss: {avg_train_stats['value_loss']:.4f}")
                logger.info(f"  Entropy Loss: {avg_train_stats['entropy_loss']:.4f}")
                logger.info(f"  Reward: {avg_train_stats['reward']:.4f}")
                logger.info(f"  Steps: {avg_train_stats['steps']:.2f}")
                #logger.info(f"  Correct Nodes: {avg_train_stats['correct_nodes']:.2f}")
                logger.info(f"  Perc Correct Nodes: {avg_train_stats['perc_correct_nodes']:.2f}")
                logger.info(f"  Perc Answer Nodes Reached: {avg_train_stats['perc_answer_nodes_reached']:.2f}")
                logger.info(f"  Perc Any Answer Node Reached: {avg_train_stats['perc_any_answer_node_reached']:.2f}")
                logger.info(f"  Visited Nodes: {avg_train_stats['visited_nodes']:.2f}")
                logger.info(f"  Visited Edges: {avg_train_stats['visited_edges']:.2f}")
                logger.info(f"  Avg Entropy: {avg_entropy:.4f}")

        logger.info(f"Epoch {epoch} stats")
        
        # Compute average training stats
        if train_stats:
            avg_train_stats = {
                k: sum(s[k] for s in train_stats) / len(train_stats)
                for k in train_stats[0].keys() if k != 'entropy'
            }
            avg_entropy = sum([sum(s['entropy']) for s in train_stats]) / sum([len(s['entropy']) for s in train_stats])
            logger.info(f"Epoch {epoch} - Training:")
            logger.info(f"  Loss: {avg_train_stats['loss']:.4f}")
            logger.info(f"  Policy Loss: {avg_train_stats['policy_loss']:.4f}")
            logger.info(f"  Value Loss: {avg_train_stats['value_loss']:.4f}")
            logger.info(f"  Entropy Loss: {avg_train_stats['entropy_loss']:.4f}")
            logger.info(f"  Reward: {avg_train_stats['reward']:.4f}")
            logger.info(f"  Steps: {avg_train_stats['steps']:.2f}")
            #logger.info(f"  Correct Nodes: {avg_train_stats['correct_nodes']:.2f}")
            logger.info(f"  Perc Correct Nodes: {avg_train_stats['perc_correct_nodes']:.2f}")
            logger.info(f"  Perc Answer Nodes Reached: {avg_train_stats['perc_answer_nodes_reached']:.2f}")
            logger.info(f"  Perc Any Answer Node Reached: {avg_train_stats['perc_any_answer_node_reached']:.2f}")
            logger.info(f"  Visited Nodes: {avg_train_stats['visited_nodes']:.2f}")
            logger.info(f"  Visited Edges: {avg_train_stats['visited_edges']:.2f}")
            logger.info(f"  Avg Entropy: {avg_entropy:.4f}")
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

                if q_nodes is None or len(q_nodes) == 0 or a_nodes is None or len(a_nodes) == 0:
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
                else:
                    nodes_in_subgraph_percentage = 0

                # Check if any target nodes were reached
                success = any(node in visited_nodes for node in a_nodes)
                
                val_stats.append({
                    'success': 1.0 if success else 0.0,
                    'nodes': len(visited_nodes),
                    'edges': len(visited_edges),
                    'nodes_in_subgraph_percentage': nodes_in_subgraph_percentage
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
            logger.info(f"  Avg Nodes in Subgraph: {avg_val_stats['nodes_in_subgraph_percentage']:.2f}")
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


def pretrain_classifier(sl_classifier, sl_optimizer, dataset, train_indices, val_indices, device, logger, args, checkpoint_dir):
    for epoch in range(args.sl_epochs):
        epoch_loss = 0

        for i in tqdm(range(len(train_indices))):
            idx = train_indices[i]

            sample = dataset[idx]
            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue
            trajectory = build_pretrain_trajectory(sample, device, directed=args.directed, triple_graph=args.triple_graph)
            if trajectory is None:
                logger.warning(f"Trajectory {idx} is None")
                continue
            
            sl_optimizer.zero_grad()

            y_pred, _, _, _, _ = sl_classifier(x_ = trajectory['x'].float(), 
                                        edge_index = trajectory['edge_index'].long(), 
                                        edge_attr = trajectory['edge_attr'].float(), 
                                        question_embeddings = trajectory['question_embeddings'], 
                                        question_mask = trajectory['question_mask']
                                        )
            
            y = trajectory['y']

            pos_weight_value = y.shape[0] / max(float(y.sum()), 1.0)
            pos_w = torch.tensor(pos_weight_value, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction='sum')  # w0 is implicitly 1
            loss = criterion(y_pred.squeeze(-1), y.float()) / y.shape[0]

            loss.backward()
            sl_optimizer.step()
            epoch_loss += loss.item()
            
        logger.info(f"Epoch {epoch} loss: {epoch_loss}")

        sl_classifier.eval()
        
        tp, fp, tn, fn = 0, 0, 0, 0
        total_samples = 0
        for idx in val_indices:
            sample = dataset[idx]

            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue
            trajectory = build_pretrain_trajectory(sample, device, directed=args.directed, triple_graph=args.triple_graph)
            if trajectory is None:
                logger.warning(f"Trajectory {idx} is None")
                continue

            y_pred, _, _, _, _ = sl_classifier(x_ = trajectory['x'].float(), 
                                        edge_index = trajectory['edge_index'].long(), 
                                        edge_attr = trajectory['edge_attr'].float(), 
                                        question_embeddings = trajectory['question_embeddings'], 
                                        question_mask = trajectory['question_mask'])
            
            y = trajectory['y']

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

        sl_classifier.train()

    torch.save(sl_classifier.state_dict(), checkpoint_dir / "SL_classifier.pt")

def train_value_head(policy_net, test_policy_net, optimizer, dataset, train_indices, val_indices, device, logger, args, checkpoint_dir):
    """
    Train the value head of policy_net using trajectories collected by test_policy_net.
    Only the value head is optimized to predict the (normalized) advantage.
    """
    trainer = RetrievalTrainer(
        test_policy_net,
        optimizer,
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

    for epoch in range(args.sl_epochs*10):
        buffer = []
        logger.info(f"Epoch {epoch} - Collecting {args.ppo_batch_size} trajectories for value head training")

        # 1. Collect batch_size trajectories using test_policy_net
        for _ in range(args.ppo_batch_size):
            idx = random.choice(train_indices)
            sample = dataset[idx]
            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue
            trajectory, _ = trainer._collect_trajectory(sample, test_policy_net, device, teacher_bias=1.0, max_steps=args.max_steps, directed=args.directed, triple_graph=args.triple_graph, epsilon=0.5)
            if trajectory is not None:
                buffer.append(trajectory)

        # 2. Flatten all steps from all trajectories into a list of (state, normalized_advantage) tuples
        all_steps = []
        all_advantages = []
        for traj in buffer:
            returns = traj.get('returns', None)
            advantages = traj.get('advantages', None)
            if returns is None:
                # Compute returns if not present
                rewards = traj['rewards']
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + 0.99 * G  # gamma = 0.99
                    returns.insert(0, G)
            if advantages is None:
                # Compute advantages as returns (since only value head is trained)
                advantages = returns
            for state, adv in zip(traj['states'], advantages):
                all_steps.append((state, adv))
                all_advantages.append(adv)

        # Normalize all advantages
        if all_advantages:
            adv_tensor = torch.tensor(all_advantages, dtype=torch.float, device=device)
            mean_adv = adv_tensor.mean()
            std_adv = adv_tensor.std() + 1e-8
            normalized_advantages = ((adv_tensor - mean_adv) / std_adv).tolist()
            # Update all_steps with normalized advantages
            all_steps = [
                (state, norm_adv)
                for (state, _), norm_adv in zip(all_steps, normalized_advantages)
            ]

        if not all_steps:
            logger.warning("No valid steps collected for value head training.")
            continue

        # 3. Randomly sample k steps from all_steps
        k = min(args.ppo_minibatch_size, len(all_steps))
        sampled_steps = random.sample(all_steps, k)

        # 4. Train policy_net's value head on these k steps
        policy_net.train()
        optimizer.zero_grad()
        value_losses = []
        for state, target_advantage in sampled_steps:
            # Forward pass through policy_net to get value prediction
            with torch.no_grad():
                x = state['x'].float().to(device).requires_grad_(True)
                edge_index = state['edge_index'].long().to(device)
                edge_attr = state['edge_attr'].float().to(device).requires_grad_(True)
                question_embeddings = state['question_embeddings'].to(device).requires_grad_(True) if state['question_embeddings'] is not None else None
                subgraph_mask = state['subgraph_mask']
                action_mask = state['action_mask']
                action_bias = state['action_bias']
                question_mask = state['question_mask']

            # Only optimize the value head
            _, value_pred, _, _ = policy_net(
                x, edge_index, edge_attr, question_embeddings,
                subgraph_mask=subgraph_mask,
                action_mask=action_mask,
                action_bias=action_bias,
                question_mask=question_mask,
                use_checkpoint=True
            )
            logger.info(f"Value pred: {value_pred.squeeze()}")
            logger.info(f"Target: {target_advantage}")
            target = torch.tensor(target_advantage, dtype=torch.float, device=device)
            #value_loss = F.mse_loss(value_pred.squeeze(), target)
            value_loss = F.smooth_l1_loss(value_pred.squeeze(), target)
            value_losses.append(value_loss)

        if value_losses:
            before = {n: p.detach().clone() for n, p in policy_net.named_parameters()}
            total_value_loss = torch.stack(value_losses).mean()
            total_value_loss.backward()

            grad_norm_custom_gcn = 0.0
            for p in policy_net.custom_GCN.parameters():
                if p.grad is not None:
                    grad_norm_custom_gcn += p.grad.data.norm(2).item() ** 2
            grad_norm_custom_gcn = grad_norm_custom_gcn ** 0.5

            logger.info(f"Epoch {epoch} - custom_GCN grad norm: {grad_norm_custom_gcn:.6f}")
            optimizer.step()
            deltas = {n: (p - before[n]) for n, p in policy_net.named_parameters()}
            logger.info(f"Epoch {epoch} - Value head loss: {total_value_loss.item():.6f}")
            tot_l2   = torch.sqrt(sum(d.pow(2).sum() for d in deltas.values()))
            max_abs  = max(d.abs().max() for d in deltas.values())
            logger.info(f"Epoch {epoch} - L2 norm: {tot_l2:.6f}, max abs: {max_abs:.6f}")
            logger.info(f"Relative Step Size: {max_abs / (max(1e-8, max(p.abs().max() for p in policy_net.parameters())))}")
        else:
            logger.warning(f"Epoch {epoch} - No value loss computed (empty buffer or no valid steps)")

        # 5. Empty the buffer
        buffer.clear()

def train_value_head_test(policy_net, optimizer, dataset, train_indices, val_indices, device, logger, args, checkpoint_dir):
    """
    Train the value head of policy_net using trajectories collected by test_policy_net.
    Only the value head is optimized to predict the (normalized) advantage.
    """
    trainer = RetrievalTrainer(
        policy_net,
        optimizer,
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

    for epoch in range(args.sl_epochs*10):
        buffer = []
        logger.info(f"Epoch {epoch} - Collecting {args.ppo_batch_size} trajectories for value head training")

        # 1. Collect batch_size trajectories using test_policy_net
        for _ in range(args.ppo_batch_size):
            idx = random.choice(train_indices)
            sample = dataset[idx]
            if sample is None:
                logger.warning(f"Sample {idx} is None")
                continue
            trajectory, _ = trainer._collect_trajectory(sample, policy_net, device, teacher_bias=1.0, max_steps=args.max_steps, directed=args.directed, triple_graph=args.triple_graph, epsilon=0.0)
            if trajectory is not None:
                buffer.append(trajectory)

        # 2. Flatten all steps from all trajectories into a list of (state, normalized_advantage) tuples
        all_steps = []
        all_advantages = []
        for traj in buffer:
            returns = traj.get('returns', None)
            advantages = traj.get('advantages', None)
            if returns is None:
                # Compute returns if not present
                rewards = traj['rewards']
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + 0.99 * G  # gamma = 0.99
                    returns.insert(0, G)
            if advantages is None:
                # Compute advantages as returns (since only value head is trained)
                advantages = returns
            for state, adv in zip(traj['states'], advantages):
                all_steps.append((state, adv))
                all_advantages.append(adv)

        # Normalize all advantages
        if all_advantages:
            adv_tensor = torch.tensor(all_advantages, dtype=torch.float, device=device)
            mean_adv = adv_tensor.mean()
            std_adv = adv_tensor.std() + 1e-8
            normalized_advantages = ((adv_tensor - mean_adv) / std_adv).tolist()
            # Update all_steps with normalized advantages
            all_steps = [
                (state, norm_adv)
                for (state, _), norm_adv in zip(all_steps, normalized_advantages)
            ]

        if not all_steps:
            logger.warning("No valid steps collected for value head training.")
            continue

        # 3. Randomly sample k steps from all_steps
        k = min(args.ppo_minibatch_size, len(all_steps))
        sampled_steps = random.sample(all_steps, k)

        # 4. Train policy_net's value head on these k steps
        policy_net.train()
        optimizer.zero_grad()
        value_losses = []
        for state, target_advantage in sampled_steps:
            # Forward pass through policy_net to get value prediction
            with torch.no_grad():
                x = state['x'].float().to(device).requires_grad_(True)
                edge_index = state['edge_index'].long().to(device)
                edge_attr = state['edge_attr'].float().to(device).requires_grad_(True)
                question_embeddings = state['question_embeddings'].to(device).requires_grad_(True) if state['question_embeddings'] is not None else None
                subgraph_mask = state['subgraph_mask']
                action_mask = state['action_mask']
                action_bias = state['action_bias']
                question_mask = state['question_mask']

            # Only optimize the value head
            _, value_pred, _, _ = policy_net(
                x, edge_index, edge_attr, question_embeddings,
                subgraph_mask=subgraph_mask,
                action_mask=action_mask,
                action_bias=action_bias,
                question_mask=question_mask,
                use_checkpoint=True
            )
            logger.info(f"Value pred: {value_pred.squeeze()}")
            logger.info(f"Target: {target_advantage}")
            target = torch.tensor(target_advantage, dtype=torch.float, device=device)
            value_loss = F.mse_loss(value_pred.squeeze(), target)
            value_losses.append(value_loss)

        if value_losses:
            before = {n: p.detach().clone() for n, p in policy_net.named_parameters()}
            total_value_loss = torch.stack(value_losses).mean()
            total_value_loss.backward()

            grad_norm_critic = 0.0
            for p in policy_net.critic.parameters():
                if p.grad is not None:
                    grad_norm_critic += p.grad.data.norm(2).item() ** 2
            grad_norm_critic = grad_norm_critic ** 0.5

            logger.info(f"Epoch {epoch} - critic grad norm: {grad_norm_critic:.6f}")
            optimizer.step()
            deltas = {n: (p - before[n]) for n, p in policy_net.named_parameters()}
            logger.info(f"Epoch {epoch} - Value head loss: {total_value_loss.item():.6f}")
            tot_l2   = torch.sqrt(sum(d.pow(2).sum() for d in deltas.values()))
            max_abs  = max(d.abs().max() for d in deltas.values())
            logger.info(f"Epoch {epoch} - L2 norm: {tot_l2:.6f}, max abs: {max_abs:.6f}")
            logger.info(f"Relative Step Size: {max_abs / (max(1e-8, max(p.abs().max() for p in policy_net.parameters())))}")
        else:
            logger.warning(f"Epoch {epoch} - No value loss computed (empty buffer or no valid steps)")

        # 5. Empty the buffer
        buffer.clear()

def train_policy():
    pass

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    main()