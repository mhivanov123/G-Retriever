import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset, concatenate_datasets
from torch_geometric.data import Data
from src.utils.lm_modeling import load_model, load_text2embedding


model_name = 'gte'
path = 'dataset/cwq'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

HF_DIR = "hf"
HF_MODELS_DIR = os.path.join(HF_DIR, "models")
HF_DATASETS_DIR = os.path.join(HF_DIR, "datasets")
os.makedirs(HF_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_DIR, exist_ok=True)
cwq_dataset_path = os.path.join(HF_DATASETS_DIR, "RoG-cwq")



def step_one():
    dataset = load_dataset("rmanluo/RoG-cwq")
    #dataset = load_dataset(webqsp_dataset_path)
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

    dataset = load_dataset("rmanluo/RoG-cwq")
    #dataset = load_dataset(webqsp_dataset_path)

    train_indices = np.arange(len(dataset['train']))
    val_indices = np.arange(len(dataset['validation'])) + len(dataset['train'])
    test_indices = np.arange(len(dataset['test'])) + len(dataset['train']) + len(dataset['validation'])

    # Fix bug: remove the indices of the empty graphs from the val indices
    #val_indices = [i for i in val_indices if i != 2937]

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
    dataset = load_dataset("rmanluo/RoG-cwq")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    questions = [i['question'] for i in dataset]

    model, tokenizer, device = load_model[model_name]()
    print(device)
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

    # 1. Pool all unique node_attr and edge_attr
    print('Pooling unique node and edge attributes...')
    all_node_attrs = set()
    all_edge_attrs = set()
    num_graphs = len(dataset)
    for index in range(num_graphs):
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        all_node_attrs.update(nodes['node_attr'].fillna("").tolist())
        all_edge_attrs.update(edges['edge_attr'].fillna("").tolist())

    all_node_attrs = sorted(list(all_node_attrs))
    all_edge_attrs = sorted(list(all_edge_attrs))
    node_attr2idx = {attr: idx for idx, attr in enumerate(all_node_attrs)}
    edge_attr2idx = {attr: idx for idx, attr in enumerate(all_edge_attrs)}

    # 2. Filter out machine codes for embedding
    def is_machine_code(attr):
        return isinstance(attr, str) and (attr.startswith('m.') or attr.startswith('g.'))

    # Prepare lists for embedding and mapping
    node_attrs_to_embed = [attr for attr in all_node_attrs if not is_machine_code(attr)]
    edge_attrs_to_embed = [attr for attr in all_edge_attrs if not is_machine_code(attr)]

    print('Encoding all unique node attributes (excluding machine codes)...')
    node_attr_embs_raw = text2embedding(model, tokenizer, device, node_attrs_to_embed)
    print('Encoding all unique edge attributes (excluding machine codes)...')
    edge_attr_embs_raw = text2embedding(model, tokenizer, device, edge_attrs_to_embed)

    # Get embedding dimension
    emb_dim = node_attr_embs_raw.shape[1] if len(node_attr_embs_raw.shape) > 1 else 1
    zero_emb = torch.zeros(emb_dim, dtype=node_attr_embs_raw.dtype, device=node_attr_embs_raw.device)

    # Build full embedding matrices, inserting zeros for machine codes
    node_attr_embs = []
    node_embed_lookup = {}
    j = 0
    for attr in all_node_attrs:
        if is_machine_code(attr):
            node_attr_embs.append(zero_emb)
        else:
            node_attr_embs.append(node_attr_embs_raw[j])
            j += 1
    node_attr_embs = torch.stack(node_attr_embs)

    edge_attr_embs = []
    j = 0
    for attr in all_edge_attrs:
        if is_machine_code(attr):
            edge_attr_embs.append(zero_emb)
        else:
            edge_attr_embs.append(edge_attr_embs_raw[j])
            j += 1
    edge_attr_embs = torch.stack(edge_attr_embs)

    # 3. For each subgraph, map node/edge attributes to their embedding indices and reconstruct tensors
    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for index in tqdm(range(num_graphs)):
        if os.path.exists(f'{path_graphs}/{index}.pt'):
            continue
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        nodes['node_attr'] = nodes['node_attr'].fillna("")
        if len(nodes) == 0:
            print(f'Empty graph at index {index}')
            continue

        # Map node_attr and edge_attr to their embedding indices
        node_indices = nodes['node_attr'].map(node_attr2idx).tolist()
        edge_indices = edges['edge_attr'].map(edge_attr2idx).tolist()

        # Gather embeddings
        x = node_attr_embs[node_indices]
        edge_attr = edge_attr_embs[edge_indices]
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(pyg_graph, f'{path_graphs}/{index}.pt')


if __name__ == '__main__':
    #step_one()
    step_two()
    generate_split()