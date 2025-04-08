import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding
from collections import defaultdict

model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'
path_graphs_triple = f'{path}/graphs_triple'
path_q_idx = f'{path}/q_idx_triple'

HF_DIR = "hf"
HF_MODELS_DIR = os.path.join(HF_DIR, "models")
HF_DATASETS_DIR = os.path.join(HF_DIR, "datasets")
os.makedirs(HF_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_DIR, exist_ok=True)
webqsp_dataset_path = os.path.join(HF_DATASETS_DIR, "RoG-webqsp")



def step_one():
    #dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = load_dataset(webqsp_dataset_path)
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes.to_csv(f'{path_nodes}/{i}.csv', index=False)
        edges.to_csv(f'{path_edges}/{i}.csv', index=False)


def generate_split():

    #dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = load_dataset(webqsp_dataset_path)

    train_indices = np.arange(len(dataset['train']))
    val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
    test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

    # Fix bug: remove the indices of the empty graphs from the val indices
    val_indices = [i for i in val_indices if i != 2937]

    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))


def step_two():
    print('Loading dataset...')
    #dataset = load_dataset("rmanluo/RoG-webqsp")
    
    dataset = load_dataset(webqsp_dataset_path)
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    '''questions = [i['question'] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    print(device)
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')'''

    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    os.makedirs(path_graphs_triple, exist_ok=True)
    os.makedirs(path_q_idx, exist_ok=True)
    for index in tqdm(range(len(dataset))):

        if not os.path.exists(f'{path_graphs}/{index}.pt') or os.path.exists(f'{path_graphs_triple}/{index}.pt'):
            continue

        # nodes
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        #edges = pd.read_csv(f'{path_edges}/{index}.csv')

        

        node_dict = {str(row['node_attr']).lower(): int(row['node_id']) for _, row in nodes.iterrows()}

        '''nodes['node_attr'] = nodes['node_attr'].fillna("")
        #nodes.node_attr.fillna("", inplace=True)
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')'''

        graph = torch.load(f'{path_graphs}/{index}.pt')

        triple_graph, triple_q_idx = convert_to_triple_node_graph(graph, [node_dict[str(q_node).lower()] for q_node in dataset[index]['q_entity']])
        torch.save(triple_graph, f'{path_graphs_triple}/{index}.pt')
        torch.save(triple_q_idx, f'{path_q_idx}/{index}.pt')

def convert_to_triple_node_graph(graph, q_idx):
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
        #TODO: Modify feature vector for self-loop. 
        if q_idx:
            zero_feature_vector = torch.zeros(triple_feature_dim, device=graph.x.device)
            for i, node_id in enumerate(q_idx):
                #TODO: Modify feature vector for self-loop. 
                feature_vector = torch.cat([graph.x[node_id], torch.zeros(graph.edge_attr.size(1), device=graph.x.device), graph.x[node_id]], dim=0)    

                self_loop_idx = num_original_edges + i  
                self_loop_triple_indices.append(self_loop_idx)
                self_loop_features_list.append(feature_vector)
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

if __name__ == '__main__':
    #step_one()
    step_two()
    #generate_split()
