import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst, retrieval_via_shortest_paths, get_bfs_supervision
import sys
import json
from torch_geometric.data import Data
from collections import defaultdict
model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

path_graphs_triple = f'{path}/graphs_triple'
path_q_idx = f'{path}/q_idx_triple'

true_cached_graph = f'{path}/true_cached_graphs'
true_cached_desc = f'{path}/true_cached_desc'

subGRAG_cached_graph = f'{path}/SubGRAG_cached_graphs'
subGRAG_cached_desc = f'{path}/SubGRAG_cached_desc'

bfs_supervision_dir = f'{path}/bfs_supervision'

HF_DIR = "hf"
HF_MODELS_DIR = os.path.join(HF_DIR, "models")
HF_DATASETS_DIR = os.path.join(HF_DIR, "datasets")
os.makedirs(HF_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_DIR, exist_ok=True)
webqsp_dataset_path = os.path.join(HF_DATASETS_DIR, "RoG-webqsp")




class WebQSPDataset(Dataset):
    def __init__(self, directed = False, triple = False):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        #dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        dataset = datasets.load_dataset(webqsp_dataset_path)
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')
        self.rewritten_questions = json.load(open(f'{path}/webqsp_rewrites.json'))
        self.rewritten_questions_embs = torch.load(f'{path}/webqsp_rewrites_embs.pt')
        self.triple = triple
        self.directed = directed

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        # Convert nodes dataframe to dictionary mapping text to node id
        nodes = {str(row['node_attr']).lower(): int(row['node_id']) for _, row in nodes.iterrows()}
        id_to_node = {v: k for k, v in nodes.items()}
        data = self.dataset[index]
        question = data["question"] + '?' if data["question"][-1] != '?' else data["question"]
        graph = torch.load(f'{path_graphs}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()
        q_entity = [str(q_node).lower() for q_node in data['q_entity']]
        a_entity = [str(a_node).lower() for a_node in data['a_entity']]
        #print(q_entity, a_entity)
        q_idx = [nodes[q_node] for q_node in q_entity if q_node in nodes]
        a_idx = [nodes[a_node] for a_node in a_entity if a_node in nodes]
        #print(q_idx)
        #print(a_idx)
        bfs = torch.load(f'{bfs_supervision_dir}/{index}.pt')
        shortest_path_nodes = extract_node_ids(f'{true_cached_desc}/{index}.txt')
        q_emb = self.q_embs[index]
        rewritten_question = self.rewritten_questions[str(index)]['q'][0]
        rewritten_question_emb = self.rewritten_questions_embs[index][:len(rewritten_question)]

        '''edge_id_to_row = {}
        for i, row in edges.iterrows():
            edge_id_to_row[i] = (row['src'], row['edge_attr'], row['dst'])'''
        
        edge_id_to_row = {i: (int(row['src']), str(row['edge_attr']), int(row['dst'])) for i, row in edges.iterrows()}

        if not self.directed:
            graph.edge_index = torch.cat([graph.edge_index, graph.edge_index.flip(0)], dim=1)
            graph.edge_attr = torch.cat([graph.edge_attr, graph.edge_attr], dim=0)
        
        '''if self.triple:
            triple_graph = torch.load(f'{path_graphs_triple}/{index}.pt')
            triple_q_idx = torch.load(f'{path_q_idx}/{index}.pt')

            triple_shortest_path_nodes = self.convert_to_triple_shortest_path_nodes(graph, shortest_path_nodes)
        
            triple_a_idx = self.convert_to_triple_a_idx(a_idx, graph, triple_shortest_path_nodes)

            for node, triple_node in zip(q_idx, triple_q_idx):
                edge_id_to_row[triple_node] = (node, 'self_loop', node)

            graph = triple_graph
            a_idx = triple_a_idx
            q_idx = triple_q_idx
            shortest_path_nodes = triple_shortest_path_nodes'''
        
        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
            'q_entity': q_entity,
            'a_entity': a_entity,
            'q_idx': q_idx,
            'a_idx': a_idx,
            'bfs': bfs,
            'shortest_path_nodes': shortest_path_nodes,
            'node_dict': nodes,
            'id_to_node': id_to_node,
            'edge_id_to_row': edge_id_to_row,
            'q_emb': q_emb,
            'rewritten_question': rewritten_question,
            'rewritten_question_emb': rewritten_question_emb,
        }
    
    def convert_to_triple_node_graph(self, graph, q_idx):
        """
        Converts a standard graph into a new graph where each node represents
        an original edge (triple: src_node, edge, dst_node). Adds special
        self-loop triple nodes for nodes in q_idx. Edges are created only
        from triple i to triple j if the destination node of triple i is the
        source node of triple j (tail-to-head connection).

        Args:
            graph (torch_geometric.data.Data): The original graph object.
                Must have graph.x, graph.edge_index, and graph.edge_attr.
            q_idx (list): List of node indices from the original graph that form the question.

        Returns:
            tuple(torch_geometric.data.Data, list):
                - A new graph where nodes are triples (including self-loop triples).
                - A list of indices corresponding to the added self-loop triple nodes.
                Returns (None, []) if the original graph is invalid or has no edges.
        """
        if graph.edge_index is None or graph.edge_attr is None or graph.x is None:
            print("Warning: Original graph missing x, edge_index, or edge_attr. Cannot create triple-node graph.")
            return None, []
        # Determine feature dimension early, needed for empty graph case too
        triple_feature_dim = graph.x.size(1) * 2 + graph.edge_attr.size(1) if graph.edge_attr is not None else graph.x.size(1) * 2

        if graph.edge_index.size(1) == 0 and not q_idx: # Allow creation if only q_idx exists for self-loops
             print("Warning: Original graph has no edges and no q_idx. Cannot create triple-node graph.")
             # Return an empty graph structure (without edge_attr)
             return Data(x=torch.empty((0, triple_feature_dim), device=graph.x.device),
                         edge_index=torch.empty((2, 0), dtype=torch.long, device=graph.x.device)), []


        num_original_edges = graph.edge_index.size(1) if graph.edge_index is not None else 0
        if num_original_edges > 0:
            src_nodes = graph.edge_index[0]
            dst_nodes = graph.edge_index[1]
            # 1a. Create features for the regular "triple" nodes
            src_node_features = graph.x[src_nodes]
            dst_node_features = graph.x[dst_nodes]
            edge_features = graph.edge_attr
            regular_node_features = torch.cat([src_node_features, edge_features, dst_node_features], dim=1)
            # triple_feature_dim is already calculated above
        else:
            # No regular edges, features will only be self-loops if q_idx exists
            regular_node_features = torch.empty((0, triple_feature_dim), device=graph.x.device)


        # 1b. Create zero features for the self-loop "triple" nodes
        self_loop_triple_indices = []
        self_loop_features_list = []
        q_node_to_self_loop_idx = {}
        if q_idx:
            zero_feature_vector = torch.zeros(triple_feature_dim, device=graph.x.device)
            for i, node_id in enumerate(q_idx):
                self_loop_idx = num_original_edges + i
                self_loop_triple_indices.append(self_loop_idx)
                self_loop_features_list.append(zero_feature_vector)
                q_node_to_self_loop_idx[node_id] = self_loop_idx # Map original q_node to its self-loop triple index

            self_loop_features = torch.stack(self_loop_features_list)
            # Combine regular and self-loop features
            new_node_features = torch.cat([regular_node_features, self_loop_features], dim=0)
        else:
            new_node_features = regular_node_features # No self-loops to add

        # 2. Create edges between the new "triple" nodes (including self-loops)
        new_edge_list = []
        # Use mappings for efficiency: original_node_id -> list of triple_node_indices involving it
        node_to_outgoing_triple_indices = defaultdict(list)
        node_to_incoming_triple_indices = defaultdict(list)
        if num_original_edges > 0:
            for i in range(num_original_edges):
                node_to_outgoing_triple_indices[src_nodes[i].item()].append(i)
                node_to_incoming_triple_indices[dst_nodes[i].item()].append(i)

        # Directed: Connect triple i to triple j if dst(i) == src(j)
        # Also connect self-loops appropriately
        original_node_ids = set(node_to_incoming_triple_indices.keys()) | set(node_to_outgoing_triple_indices.keys()) | set(q_idx)

        for node_id in original_node_ids:
            incoming_triples = node_to_incoming_triple_indices[node_id]
            outgoing_triples = node_to_outgoing_triple_indices[node_id]
            self_loop_idx = q_node_to_self_loop_idx.get(node_id) # Get self-loop index if this node is in q_idx

            # Connect regular incoming to regular outgoing
            for in_idx in incoming_triples:
                for out_idx in outgoing_triples:
                    new_edge_list.append([in_idx, out_idx])

            # Connect involving self-loops if present
            if self_loop_idx is not None:
                # Incoming -> SelfLoop
                for in_idx in incoming_triples:
                    new_edge_list.append([in_idx, self_loop_idx])
                # SelfLoop -> Outgoing
                for out_idx in outgoing_triples:
                    new_edge_list.append([self_loop_idx, out_idx])
                # SelfLoop -> SelfLoop (added for completeness, might be redundant depending on use case)
                new_edge_list.append([self_loop_idx, self_loop_idx])


        if not new_edge_list:
            new_edge_index = torch.empty((2, 0), dtype=torch.long, device=graph.x.device)
        else:
            new_edge_index = torch.tensor(new_edge_list, dtype=torch.long, device=graph.x.device).t().contiguous()
            # Remove duplicate edges if any
            new_edge_index = torch.unique(new_edge_index, dim=1)

        # 3. Create edge attributes (all zeros) for the new edges - REMOVED
        # num_new_edges = new_edge_index.size(1)
        # new_edge_attr = torch.zeros((num_new_edges, triple_feature_dim), device=graph.x.device)


        # 4. Create the new graph Data object (without edge_attr)
        new_graph = Data(x=new_node_features, edge_index=new_edge_index)
        # Optionally copy other attributes if needed
        # new_graph.graph_id = graph.graph_id

        return new_graph, self_loop_triple_indices

    def convert_to_triple_shortest_path_nodes(self, graph, shortest_path_nodes):
        """
        Convert node indices from the original graph to triple indices in the line graph.
        
        Args:
            shortest_path_nodes: List of node indices from the original graph that form a shortest path
            
        Returns:
            List of triple indices (edge indices in the original graph) that connect nodes in the shortest path
        """
        if not shortest_path_nodes or len(shortest_path_nodes) < 2:
            return []  # Need at least two nodes to form a path with edges
        
        # Find all edges (triples) where both endpoints are in the shortest path nodes
        triple_indices = []
        shortest_path_nodes_set = set(shortest_path_nodes)
        
        # Iterate through the original edges and check if both endpoints are in the shortest path
        for i, (src, dst) in enumerate(zip(graph.edge_index[0], graph.edge_index[1])):
            src_node = src.item()
            dst_node = dst.item()
            
            if src_node in shortest_path_nodes_set and dst_node in shortest_path_nodes_set:
                triple_indices.append(i)
        
        return triple_indices

    def convert_to_triple_a_idx(self, a_idx, graph, triple_shortest_path_nodes):
        """
        Convert answer node indices from the original graph to triple indices in the line graph.
        
        Args:
            a_idx: List of node indices from the original graph that form the answer"""
        
        triple_a_idx = []
        for a in a_idx:
            for i, triple_idx in enumerate(triple_shortest_path_nodes):
                if graph.edge_index[1][triple_idx] == a or graph.edge_index[0][triple_idx] == a:
                    triple_a_idx.append(triple_idx)
        return triple_a_idx
    
    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}
    
    def is_connected_component(self, triple_graph, triple_shortest_path_nodes):
        """
        Checks if the triple shortest path nodes form a connected component.
        
        Args:
            triple_graph: The triple node graph
            triple_shortest_path_nodes: List of triple indices that should form a connected path
            
        Returns:
            Boolean indicating if the nodes form a connected component
        """
        if not triple_shortest_path_nodes or len(triple_shortest_path_nodes) <= 1:
            return True  # 0 or 1 node is always connected
        
        # Create a subgraph with only edges between triple_shortest_path_nodes
        nodes_set = set(triple_shortest_path_nodes)
        adj_list = defaultdict(list)
        
        for i, (src, dst) in enumerate(zip(triple_graph.edge_index[0], triple_graph.edge_index[1])):
            src_node = src.item()
            dst_node = dst.item()
            if src_node in nodes_set and dst_node in nodes_set:
                adj_list[src_node].append(dst_node)
        
        # Run BFS from the first node
        start_node = triple_shortest_path_nodes[0]
        visited = set([start_node])
        queue = [start_node]
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all nodes were visited
        return len(visited) == len(nodes_set)

def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)

    os.makedirs(true_cached_desc, exist_ok=True)
    os.makedirs(true_cached_graph, exist_ok=True)

    os.makedirs(bfs_supervision_dir, exist_ok=True)
    #dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.load_dataset(webqsp_dataset_path)
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    
    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        '''if os.path.exists(f'{cached_graph}/{index}.pt') and os.path.exists(f'{true_cached_graph}/{index}.txt'):
            continue'''

        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue

        graph = torch.load(f'{path_graphs}/{index}.pt')
        #q_emb = q_embs[index]
        #subg, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)
        #torch.save(subg, f'{cached_graph}/{index}.pt')
        #open(f'{cached_desc}/{index}.txt', 'w').write(desc)
        #true_subg, true_desc = retrieval_via_shortest_paths(graph, dataset[index]['q_entity'], dataset[index]['a_entity'], nodes, edges)
        #torch.save(true_subg, f'{true_cached_graph}/{index}.pt')
        #open(f'{true_cached_desc}/{index}.txt', 'w').write(true_desc)

        bfs_supervision, success = get_bfs_supervision(graph, dataset[index]['q_entity'], dataset[index]['a_entity'], nodes)
        torch.save(bfs_supervision, f'{bfs_supervision_dir}/{index}.pt')

def extract_node_ids(file_path):
    node_ids = []
    with open(file_path, 'r') as f:
        # Skip header
        next(f)
        # Read until empty line
        for line in f:
            if line.strip() == "":
                break
            node_id = int(line.split(',')[0])
            node_ids.append(node_id)
    return node_ids

if __name__ == '__main__':

    preprocess()

    dataset = WebQSPDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
