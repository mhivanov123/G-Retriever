import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst, retrieval_via_shortest_paths, get_bfs_supervision
import sys
import json
model_name = 'sbert'
path = 'dataset/webqsp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

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
    def __init__(self):
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
        q_idx = [nodes[q_node] for q_node in q_entity if q_node in nodes]
        a_idx = [nodes[a_node] for a_node in a_entity if a_node in nodes]
        bfs = torch.load(f'{bfs_supervision_dir}/{index}.pt')
        shortest_path_nodes = extract_node_ids(f'{true_cached_desc}/{index}.txt')
        q_emb = self.q_embs[index]
        rewritten_question = self.rewritten_questions[str(index)]['q'][0]
        rewritten_question_emb = self.rewritten_questions_embs[index][:len(rewritten_question)]

        '''edge_id_to_row = {}
        for i, row in edges.iterrows():
            edge_id_to_row[i] = (row['src'], row['edge_attr'], row['dst'])'''
        
        edge_id_to_row = {i: (int(row['src']), str(row['edge_attr']), int(row['dst'])) for i, row in edges.iterrows()}
        

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
            'rewritten_question_emb': rewritten_question_emb
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}
    



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
