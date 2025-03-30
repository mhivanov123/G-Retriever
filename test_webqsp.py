import torch
from src.dataset.webqsp import WebQSPDataset
from torch_geometric.data import Data


def main():
    # Initialize dataset
    dataset = WebQSPDataset()
    print(f"\nDataset size: {len(dataset)}")
    
    # Get split information
    split_indices = dataset.get_idx_split()
    print("\nSplit sizes:")
    for split, indices in split_indices.items():
        print(f"{split}: {len(indices)}")
    
    # Examine a few samples
    sample_indices = [0, 1, 2]  # Can modify these to look at different examples
    
    for idx in sample_indices:
        print(f"\n=== Sample {idx} ===")
        data = dataset[idx]
        
        # Print basic information
        print(f"Question: {data['question']}")
        print(f"Label (Answer): {data['label']}")
        print(f"Question entities: {data['q_entity']}")
        print(f"Answer entities: {data['a_entity']}")
        # Print BFS supervision information
        if 'bfs' in data:
            print("\nBFS supervision layers:")
            for i, layer in enumerate(data['bfs']):
                print(f"Layer {i}: {layer}")
        

if __name__ == "__main__":
    main() 