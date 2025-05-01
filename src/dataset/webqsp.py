import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst, retrieval_via_shortest_paths, get_bfs_supervision, shortest_path_retrieval
import pickle

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

shortest_path_nodes_dir = f'{path}/shortest_path_nodes'

subGRAG_embedding_graph = f'{path}/subGRAG_bfs_supervision'

HF_DIR = "hf"
HF_MODELS_DIR = os.path.join(HF_DIR, "models")
HF_DATASETS_DIR = os.path.join(HF_DIR, "datasets")
os.makedirs(HF_DIR, exist_ok=True)
os.makedirs(HF_MODELS_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_DIR, exist_ok=True)
#webqsp_dataset_path = os.path.join(HF_DATASETS_DIR, "RoG-webqsp")
webqsp_dataset_path = 'rmanluo/RoG-webqsp'
class WebQSPDataset(Dataset):
    def __init__(self, directed=False, triple=False):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        
        dataset = datasets.load_dataset(webqsp_dataset_path)
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')
        
        self.directed = directed
        self.triple = triple
        
        #train = pickle.load(open(f'/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/processed/train.pkl', 'rb'))
        #val = pickle.load(open(f'/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/processed/val.pkl', 'rb'))
        #test = pickle.load(open(f'/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/processed/test.pkl', 'rb'))
        #self.subgrph_rag_dataset = train + val + test

        #self.id_to_index = {item['id']: index for index, item in enumerate(self.subgrph_rag_dataset)}

        # Load embeddings with map_location to ensure they stay on CPU
        #train_dict = torch.load('/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/emb/gte-large-en-v1.5/train.pth', 
        #                       map_location=torch.device('cpu'))
        #val_dict = torch.load('/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/emb/gte-large-en-v1.5/val.pth', 
        #                     map_location=torch.device('cpu'))
        #test_dict = torch.load('/home/gridsan/mhadjiivanov/meng/SubgraphRAG/retrieve/data_files/webqsp/emb/gte-large-en-v1.5/test.pth', 
        #                      map_location=torch.device('cpu'))

        #self.subgraph_rag_emb_dict = {**train_dict, **val_dict, **test_dict}
        
        # Convert all tensors to FP16 to save memory
        #for k, v in self.subgraph_rag_emb_dict.items():
        #    for k2, v2 in v.items():
        #        if isinstance(v2, torch.Tensor):
        #            self.subgraph_rag_emb_dict[k][k2] = v2.clone().half().cpu()

    def subgraph_rag_entity_to_index(self, sample_id):
        subgraph_rag_id = self.id_to_index[sample_id]
        sub_dict = self.subgrph_rag_dataset[subgraph_rag_id]
        entity_to_index = {}
        for index, item in enumerate(sub_dict['text_entity_list']):
            entity_to_index[item.lower()] = index

        relation_to_index = {}
        for index, item in enumerate(sub_dict['relation_list']):
            relation_to_index[item.lower()] = index
        
        return entity_to_index, relation_to_index

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        sample_id = data['id']

        nodes_raw = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        # Convert nodes dataframe to dictionary mapping text to node id
        nodes = {str(row['node_attr']).lower(): int(row['node_id']) for _, row in nodes_raw.iterrows()}
        id_to_node = {v: k for k, v in nodes.items()}
        
        question = data["question"] + '?' if data["question"][-1] != '?' else data["question"]
        graph = torch.load(f'{path_graphs}/{index}.pt', map_location=torch.device('cpu'))

        #desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()
        
        q_entity = [str(q_node).lower() for q_node in data['q_entity']]
        a_entity = [str(a_node).lower() for a_node in data['a_entity']]
        q_idx = [nodes[q_node] for q_node in q_entity if q_node in nodes]
        a_idx = [nodes[a_node] for a_node in a_entity if a_node in nodes]
        shortest_path_nodes = torch.load(f'{shortest_path_nodes_dir}/{index}.pt')

        print(f"shortest_path_nodes: {shortest_path_nodes}")
        
        # Use the FP16 question embedding
        #q_emb = self.subgraph_rag_emb_dict[sample_id]['q_emb']
        q_emb = self.q_embs[index]
        
        edge_id_to_row = {i: (int(row['src']), str(row['edge_attr']), int(row['dst'])) for i, row in edges.iterrows()}

        if not self.directed:
            graph.edge_index = torch.cat([graph.edge_index, torch.flip(graph.edge_index, [0])], dim=1)
            graph.edge_attr = torch.cat([graph.edge_attr, graph.edge_attr], dim=0)

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'q_entity': q_entity,
            'a_entity': a_entity,
            'q_idx': q_idx,
            'a_idx': a_idx,
            'shortest_path_nodes': shortest_path_nodes,
            'node_dict': nodes,
            'id_to_node': id_to_node,
            'edge_id_to_row': edge_id_to_row,
            'q_emb': q_emb
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
    #os.makedirs(cached_desc, exist_ok=True)
    #os.makedirs(cached_graph, exist_ok=True)

    #os.makedirs(true_cached_desc, exist_ok=True)
    #os.makedirs(true_cached_graph, exist_ok=True)

    os.makedirs(shortest_path_nodes_dir, exist_ok=True)
    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    #dataset = datasets.load_dataset(webqsp_dataset_path)
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

        shortest_path_nodes = shortest_path_retrieval(graph, dataset[index]['q_entity'], dataset[index]['a_entity'], nodes, edges)
        torch.save(shortest_path_nodes, f'{shortest_path_nodes_dir}/{index}.pt')

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

    #preprocess()

    dataset = WebQSPDataset()

    for i in range(len(dataset)):
        data = dataset[i]

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
