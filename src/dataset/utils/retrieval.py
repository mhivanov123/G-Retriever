import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import sys


def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    # Convert tensors to full precision (FP32) and ensure they're on the same device
    q_emb = q_emb.float()  # Convert from half to float
    graph.x = graph.x.float()  # Convert graph embeddings to float too
    
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}
    
    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc


def retrieval_via_shortest_paths(graph, q_nodes, a_nodes, textual_nodes, textual_edges):
    """
    Extract shortest paths between question and answer entities and form a subgraph.
    
    Args:
        graph: PyG Data object containing the full graph
        q_nodes: List of strings matching node text in textual_nodes['node_attr']
        a_nodes: List of strings matching node text in textual_nodes['node_attr']
        textual_nodes: DataFrame containing node text information
        textual_edges: DataFrame containing edge text information
    
    Returns:
        data: PyG Data object containing the subgraph
        desc: String description of the subgraph
    """
    
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        print("Empty textual nodes or edges", flush=True)
        sys.stdout.flush()
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Convert input nodes to lowercase
    q_nodes = [str(n).lower() for n in q_nodes]
    a_nodes = [str(n).lower() for n in a_nodes]
    
    # Map string identifiers to node indices (with lowercase comparison)
    node_to_idx = {str(row['node_attr']).lower(): idx for idx, row in textual_nodes.iterrows()}
    
    # Convert string nodes to indices, skipping any that aren't found
    q_node_indices = [node_to_idx[n] for n in q_nodes if n in node_to_idx]
    a_node_indices = [node_to_idx[n] for n in a_nodes if n in node_to_idx]
    
    print(f"q_node_indices: {q_node_indices}", flush=True)
    sys.stdout.flush()
    
    # If no valid nodes found, return full graph
    if not q_node_indices or not a_node_indices:
        print('no valid nodes found')
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Convert edge_index to adjacency list format for easier path finding
    adj_list = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.tolist()):
        if src not in adj_list:
            adj_list[src] = []
        if dst not in adj_list:
            adj_list[dst] = []
        adj_list[src].append((dst, i))
        adj_list[dst].append((src, i))  # Add reverse edge for undirected graph

    def bfs_shortest_path(start, end):
        if start == end:
            return [], []
        
        visited = {start}
        queue = [(start, [], [])]  # (node, path_nodes, path_edges)
        
        while queue:
            current, path_nodes, path_edges = queue.pop(0)
            
            for next_node, edge_idx in adj_list.get(current, []):
                if next_node == end:
                    return path_nodes + [current, next_node], path_edges + [edge_idx]
                
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path_nodes + [current], path_edges + [edge_idx]))
        
        return [], []  # No path found

    # Collect all nodes and edges in shortest paths
    selected_nodes = set()
    selected_edges = set()
    
    for q_node in q_node_indices:
        for a_node in a_node_indices:
            path_nodes, path_edges = bfs_shortest_path(q_node, a_node)
            selected_nodes.update(path_nodes)
            selected_edges.update(path_edges)

    print(selected_nodes)
    print(selected_edges)

    selected_nodes = list(selected_nodes)
    selected_edges = list(selected_edges)

    if not selected_nodes or not selected_edges:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    # Create mapping for new node indices
    mapping = {n: i for i, n in enumerate(selected_nodes)}

    # Create new graph with selected nodes and edges
    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    edge_index = graph.edge_index[:, selected_edges]
    
    # Remap node indices
    src = [mapping[i.item()] for i in edge_index[0]]
    dst = [mapping[i.item()] for i in edge_index[1]]
    edge_index = torch.LongTensor([src, dst])

    # Create description
    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    # Create new Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc

def get_bfs_supervision(graph, start_nodes, target_nodes, textual_nodes):
    """
    Extract BFS layers containing nodes and edges that are part of shortest paths to targets.
    Each layer represents valid nodes/edges at that distance from start nodes.
    
    Args:
        graph: PyG Data object containing the full graph
        start_nodes: List of starting node indices
        target_nodes: List of target node indices
        textual_nodes: DataFrame containing node text information
    
    Returns:
        bfs_layers: List of sets, where each set contains (edge_idx, node_idx) pairs
        success: Boolean indicating if any target was reached
    """
    start_nodes = [str(n).lower() for n in start_nodes]
    target_nodes = [str(n).lower() for n in target_nodes]
    
    # Map string identifiers to node indices (with lowercase comparison)
    node_to_idx = {str(row['node_attr']).lower(): idx for idx, row in textual_nodes.iterrows()}
    
    # Convert string nodes to indices, skipping any that aren't found
    start_nodes = [node_to_idx[n] for n in start_nodes if n in node_to_idx]
    target_nodes = [node_to_idx[n] for n in target_nodes if n in node_to_idx]

    adj_list = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.tolist()):
        if src not in adj_list:
            adj_list[src] = []
        if dst not in adj_list:
            adj_list[dst] = []
        adj_list[src].append((dst, i))
        #adj_list[dst].append((src, i))  # Add reverse edge for undirected graph

    def bfs_shortest_path(start, end):
        if start == end:
            return [], []
        
        visited = {start}
        queue = [(start, [], [])]  # (node, path_nodes, path_edges)
        
        while queue:
            current, path_nodes, path_edges = queue.pop(0)
            
            for next_node, edge_idx in adj_list.get(current, []):
                if next_node == end:
                    return path_nodes + [current, next_node], path_edges + [edge_idx]
                
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path_nodes + [current], path_edges + [edge_idx]))
        
        return [], []  # No path found
    

    # Collect all nodes and edges in shortest paths
    selected_nodes = set()
    selected_edges = set()
    
    # Find shortest paths from each start node to each target node
    success = False
    for start_node in start_nodes:
        for target_node in target_nodes:
            path_nodes, path_edges = bfs_shortest_path(start_node, target_node)
            if path_nodes:  # If path was found
                success = True
                selected_nodes.update(path_nodes)
                selected_edges.update(path_edges)
    
    if not success:
        return [], False

    # Build BFS layers from valid nodes/edges
    bfs_layers = []
    visited = set(start_nodes)
    current_layer = {(None, node) for node in start_nodes}  # First layer is just start nodes
    
    while current_layer:
        next_layer = set()
        
        for _, current in current_layer:
            # Get all edges from current node
            edges = (graph.edge_index[0] == current).nonzero().squeeze(-1)
            
            for edge_idx in edges:
                edge_idx = edge_idx.item()
                next_node = graph.edge_index[1, edge_idx].item()
                
                # Only include nodes and edges that are part of shortest paths
                if next_node not in visited and next_node in selected_nodes and edge_idx in selected_edges:
                    next_layer.add((edge_idx, next_node))
                    visited.add(next_node)
        
        if next_layer:
            bfs_layers.append(next_layer)
            current_layer = next_layer
        else:
            break

    return bfs_layers, success