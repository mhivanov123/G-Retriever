import os
import torch
import json
import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import logging

from src.models.retrieval_policy import RetrievalPolicy
from src.models.rl_retriever import RetrievalTrainer
from src.dataset.webqsp import WebQSPDataset

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_dir = Path(save_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze trained retrieval model')
    parser.add_argument('--checkpoint', type=str, default='./experiments/checkpoints/best_model.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run analysis on')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Analysis started. Loading model from {args.checkpoint}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best validation success rate: {checkpoint.get('val_success', 'unknown')}")
    logger.info(f"Model config: {config}")
    
    # Initialize model with the same architecture as in training
    policy_net = RetrievalPolicy(
        node_dim=config.get("node_dim", 1024),
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 4)
    ).to(device)
    
    # Load model weights
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.eval()
    logger.info("Model loaded successfully")
    
    # Initialize dataset
    dataset = WebQSPDataset()
    idx_split = dataset.get_idx_split()
    
    # Use test set for analysis
    test_indices = idx_split['test']
    logger.info(f"Loaded dataset with {len(test_indices)} test samples")
    
    # Initialize trainer for inference (without encoder)
    trainer = RetrievalTrainer(policy_net, encoder=None)
    
    # Run analysis
    analyze_model(trainer, dataset, test_indices, device, logger, args.output_dir)

def analyze_model(trainer, dataset, test_indices, device, logger, output_dir):
    """
    Run analysis on the loaded model
    """
    results = {
        'sample_id': [],
        'question': [],
        'success': [],
        'num_nodes': [],
        'num_edges': [],
        'path_length': [],
        'shortest_path_length': [],
        'answer_nodes_reached': []
    }
    
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each test sample
    progress_bar = tqdm(test_indices[:500])
    for idx in progress_bar:
        sample = dataset[idx]

        if  len(sample['shortest_path_nodes']) != 2:
            continue
        
        # Skip samples without question or answer nodes
        if 'q_idx' not in sample or 'a_idx' not in sample or len(sample['q_idx']) == 0 or len(sample['a_idx']) == 0:
            logger.warning(f"Sample {idx} has no question or answer nodes, skipping")
            continue
        
        # Get subgraph through inference
        visited_nodes, visited_edges = trainer.inference_step(
            sample=sample,
            max_steps=10  # Same as in training
        )
        
        # Check if any target nodes were reached
        success = any(node in visited_nodes for node in sample['a_idx'])
        
        # Calculate metrics
        shortest_path_length = len(sample.get('shortest_path_nodes', [])) if 'shortest_path_nodes' in sample else 0
        answer_nodes_reached = sum(1 for node in sample['a_idx'] if node in visited_nodes)
        
        # Store results
        results['sample_id'].append(idx)
        results['question'].append(sample.get('question', f"Sample {idx}"))
        results['success'].append(1 if success else 0)
        results['num_nodes'].append(len(visited_nodes))
        results['num_edges'].append(len(visited_edges))
        results['path_length'].append(len(visited_nodes))
        results['shortest_path_length'].append(shortest_path_length)
        results['answer_nodes_reached'].append(answer_nodes_reached)
        
        # Visualize the subgraph (for successful cases and a few failures)
        if success or (len(results['sample_id']) % 10 == 0):  # Visualize all successes and every 10th sample
            visualize_subgraph(sample, visited_nodes, visited_edges, output_dir, idx, logger)
        
        # Update progress bar
        progress_bar.set_description(f"Success: {sum(results['success'])}/{len(results['success'])}")
    
    # Calculate overall metrics
    total_samples = len(results['sample_id'])
    success_rate = sum(results['success']) / total_samples if total_samples > 0 else 0
    avg_nodes = sum(results['num_nodes']) / total_samples if total_samples > 0 else 0
    avg_edges = sum(results['num_edges']) / total_samples if total_samples > 0 else 0
    
    # Log results
    logger.info(f"Analysis complete on {total_samples} samples")
    logger.info(f"Success Rate: {success_rate:.4f} ({sum(results['success'])}/{total_samples})")
    logger.info(f"Average Nodes: {avg_nodes:.2f}")
    logger.info(f"Average Edges: {avg_edges:.2f}")
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'analysis_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Detailed results saved to {results_path}")
    
    # Save summary to JSON
    summary = {
        'total_samples': total_samples,
        'success_rate': success_rate,
        'avg_nodes': avg_nodes,
        'avg_edges': avg_edges,
        'success_count': sum(results['success']),
        'failure_count': total_samples - sum(results['success'])
    }
    
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")
    
    # Generate visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style('whitegrid')
        
        # Plot success rate
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Success', 'Failure'], y=[summary['success_count'], summary['failure_count']])
        plt.title('Success vs Failure Count')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'success_rate.png'))
        
        # Plot path length distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(results_df['path_length'], bins=20, kde=True)
        plt.axvline(avg_nodes, color='r', linestyle='--', label=f'Average: {avg_nodes:.2f}')
        plt.title('Path Length Distribution')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'path_length_distribution.png'))
        
        # Compare with shortest path
        valid_paths = results_df[results_df['shortest_path_length'] > 0]
        if len(valid_paths) > 0:
            plt.figure(figsize=(12, 6))
            plt.scatter(valid_paths['shortest_path_length'], valid_paths['path_length'], alpha=0.5)
            plt.plot([0, max(valid_paths['shortest_path_length'])], 
                     [0, max(valid_paths['shortest_path_length'])], 'r--', label='Optimal')
            plt.title('Model Path Length vs Shortest Path Length')
            plt.xlabel('Shortest Path Length')
            plt.ylabel('Model Path Length')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'path_comparison.png'))
        
        logger.info("Visualizations generated successfully")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def visualize_subgraph(sample, visited_nodes, visited_edges, output_dir, sample_id, logger):
    """
    Visualize the subgraph explored by the model
    
    Args:
        sample: The sample data containing graph information
        visited_nodes: List of nodes visited by the model
        visited_edges: List of edges visited by the model
        output_dir: Directory to save the visualization
        sample_id: ID of the sample for naming the output file
        logger: Logger instance
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Get node-to-text mapping if available
        node_to_dict = sample.get('id_to_node', {})
        
        # Add nodes to the graph
        for node_id in visited_nodes:
            # Get node label (use ID if no mapping available)
            node_label = str(node_id)
            if node_id in node_to_dict:
                node_info = node_to_dict[node_id]
                if isinstance(node_info, dict) and 'name' in node_info:
                    node_label = node_info['name']
                elif isinstance(node_info, str):
                    node_label = node_info
            
            # Determine node type
            node_type = 'entity'  # Default type
            if node_id in sample['q_idx']:
                node_type = 'question'
            elif node_id in sample['a_idx']:
                node_type = 'answer'
            
            # Add node with attributes
            G.add_node(node_id, label=node_label, type=node_type)
        
        # Add edges to the graph
        edge_index = sample['graph'].edge_index

        for edge_id in visited_edges:
            src = edge_index[0][edge_id].item()
            dst = edge_index[1][edge_id].item()
            
            # Get edge label if available
            edge_label = ""
            if 'edge_attr' in sample and 'edge_type' in sample:
                edge_type_idx = sample['edge_type'][edge_id].item()
                if edge_type_idx < len(sample['edge_attr']):
                    edge_label = sample['edge_attr'][edge_type_idx]
            
            # Add edge with attributes
            G.add_edge(src, dst, label=edge_label)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define node colors
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'question':
                node_colors.append('skyblue')
            elif G.nodes[node]['type'] == 'answer':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        # Define node sizes
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'question':
                node_sizes.append(800)
            elif G.nodes[node]['type'] == 'answer':
                node_sizes.append(800)
            else:
                node_sizes.append(500)
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrowsize=15, width=1.5, alpha=0.7)
        
        # Draw labels with smaller font for better readability
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes()}, 
                               font_size=8, font_weight='bold')
        
        # Add title
        question_text = sample.get('question', f"Sample {sample_id}")
        plt.title(f"Question: {question_text}\nNodes: {len(visited_nodes)}, Edges: {len(visited_edges)}")
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Question'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Answer'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Entity')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Remove axis
        plt.axis('off')
        
        # Save figure
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, f'subgraph_{sample_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved for sample {sample_id}")
        return True
    except Exception as e:
        logger.error(f"Error visualizing subgraph for sample {sample_id}: {e}")
        return False

if __name__ == "__main__":
    main() 