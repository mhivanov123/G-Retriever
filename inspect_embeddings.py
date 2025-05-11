import torch
import torch.nn.functional as F
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
from src.dataset.webqsp import WebQSPDataset
import networkx as nx
from torch_geometric.utils import to_networkx

def check_shortest_path_reachability(G, q_nodes, a_nodes, shortest_path_nodes):
    # Induce the subgraph
    subG = G.subgraph(shortest_path_nodes)
    reachable = set()
    for q in q_nodes:
        # BFS from q in the subgraph
        for a in a_nodes:
            if nx.has_path(subG, q, a):
                reachable.add(a)
    # All a_nodes must be reachable
    all_reachable = all(a in reachable for a in a_nodes)
    return all_reachable, reachable



def load_embeddings(file_path):
    embeddings = torch.load(file_path)
    return embeddings

def average_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    # Normalize each embedding (along dimension 1)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute the similarity matrix (each element is the cosine similarity)
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T

    # Create a mask to exclude diagonal elements (self-similarities)
    num_embeddings = embeddings.size(0)
    diag_mask = torch.eye(num_embeddings, dtype=torch.bool, device=embeddings.device)
    
    # Select non-diagonal elements
    non_diagonal_similarities = similarity_matrix[~diag_mask]
    
    # Compute and return the average cosine similarity
    avg_cos_sim = non_diagonal_similarities.mean()
    return avg_cos_sim

def GR_compute_similarity(id1) -> torch.Tensor:

    edges = pd.read_csv(f"/home/gridsan/mhadjiivanov/meng/G-Retriever/dataset/webqsp/edges/{id1}.csv")
    graph_edge_embs = torch.load(f"/home/gridsan/mhadjiivanov/meng/G-Retriever/dataset/webqsp/graphs/{id1}.pt").edge_attr

    edge_embs = []
    seen_edges = set()
    for idx, edge_name in enumerate(graph_edge_embs):
        if edge_name not in seen_edges:
            edge_embs.append(graph_edge_embs[idx])
            seen_edges.add(edge_name)

    edge_embs = torch.stack(edge_embs).to(torch.float32)
    print(edge_embs.shape)

    return plot_cosine_similarity_distribution(edge_embs, f"{id1}_edge_embs.png") 


def plot_cosine_similarity_distribution(embeddings: torch.Tensor, filename: str, bins: int = 50):
    """
    Plots the distribution of pairwise cosine similarities for a set of embedding vectors.
    
    Args:
        embeddings (torch.Tensor): A tensor of shape (N, D), where N is the number of embeddings
                                     and D is the embedding dimension.
        bins (int): Number of bins for the histogram.
    """
    # Normalize embeddings so each vector has a unit norm
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute the cosine similarity matrix (N x N)
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T

    # Create a mask to remove the diagonal (self similarities)
    num_embeddings = embeddings.size(0)
    diag_mask = torch.eye(num_embeddings, dtype=torch.bool, device=embeddings.device)
    
    # Get only the non-diagonal similarities
    non_diag_similarities = similarity_matrix[~diag_mask].cpu().numpy()

    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(non_diag_similarities, bins=bins, density=True)
    plt.title('Distribution of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)



if __name__ == "__main__":
    dataset = WebQSPDataset(directed=True, triple=True)

    idx_split = dataset.get_idx_split()
    train_indices = idx_split['train']
    val_indices = idx_split['val']

    two_hop = 0
    valid_2_hop = 0
    for idx in val_indices:
        sample = dataset[idx]
        if sample is None:
            continue
        q_nodes = sample['q_idx']
        a_nodes = sample['a_idx']
        shortest_path_nodes = sample['shortest_path_nodes']

        non_a_or_q_nodes = [node for node in shortest_path_nodes if node not in q_nodes and node not in a_nodes]
        if not a_nodes or not q_nodes:
            continue
        
        valid_2_hop += 1
        if non_a_or_q_nodes:
            two_hop += 1
            
        

        '''nxG = to_networkx(sample['graph'], to_undirected=True)
        all_reachable, reachable = check_shortest_path_reachability(nxG, q_nodes, a_nodes, shortest_path_nodes)
        if not all_reachable:
            print("Some a_nodes are not reachable from q_nodes using only shortest_path_nodes:", set(a_nodes) - reachable)
            

        if not a_nodes:
            print(f"No answer nodes for {idx}")
        if not q_nodes:
            print(f"No question nodes for {idx}")
        if not shortest_path_nodes:
            print(f"No shortest path nodes for {idx}") 
        if not non_a_or_q_nodes:
            print(f"No non a or q nodes for {idx}")'''
    print(f"2-hop perc: {two_hop/valid_2_hop}")
    # print(embs.shape)
    # avg_cos_sim = average_cosine_similarity(embs)
    # print(avg_cos_sim)
    # plot_cosine_similarity_distribution(embs, "gte_large_2056.png")
    # GR_avg_cos_sim = GR_compute_similarity(2056)
    # print(GR_avg_cos_sim)


    #embeddings = load_pickle("/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/processed/val.pkl")
    #print(len(embeddings))
    #print(embeddings[0]['relation_list'])
