import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm

class SubgraphGNNLSTM(nn.Module):
    """
    A module that combines GNN processing of subgraphs with LSTM sequential updates
    """
    def __init__(self, node_dim, hidden_dim=256, gnn_layers=2, lstm_hidden_dim=512):
        super().__init__()
        
        # GNN for processing subgraphs
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        # First layer takes node features
        self.gnn_layers.append(GATConv(node_dim, hidden_dim))
        self.gnn_norms.append(nn.LayerNorm(hidden_dim))
        
        # Additional GNN layers
        for _ in range(gnn_layers - 1):
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))
        
        # Global pooling to get fixed-size subgraph representation
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for sequential updates
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # Projection back to embedding space
        self.projection = nn.Linear(lstm_hidden_dim, 1024)
        
        # Initialize LSTM hidden state
        self.init_lstm_hidden = nn.Parameter(torch.zeros(1, 1, lstm_hidden_dim))
        self.init_lstm_cell = nn.Parameter(torch.zeros(1, 1, lstm_hidden_dim))
    
    def process_subgraph(self, x, edge_index, edge_attr, node_mask):
        """
        Process the current subgraph with GNN layers
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            node_mask: Boolean mask of nodes in the subgraph [num_nodes]
            
        Returns:
            Subgraph representation [hidden_dim]
        """
        # Extract subgraph
        sub_nodes = node_mask.nonzero().squeeze(-1)
        sub_edge_index, sub_edge_attr = subgraph(
            sub_nodes, 
            edge_index, 
            edge_attr=edge_attr, 
            relabel_nodes=True,
            num_nodes=x.size(0)
        )
        
        # If subgraph is empty, return zeros
        if sub_nodes.size(0) == 0 or sub_edge_index.size(1) == 0:
            return torch.zeros(1, self.gnn_layers[-1].out_channels, device=x.device)
        
        # Get node features for the subgraph
        sub_x = x[sub_nodes]
        
        # Process through GNN layers
        h = sub_x
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            h = gnn(h, sub_edge_index, sub_edge_attr)
            h = norm(h)
            h = F.relu(h)
        
        # Global mean pooling
        pooled = h.mean(dim=0, keepdim=True)
        
        # Process through MLP
        subgraph_repr = self.pool_mlp(pooled)
        
        return subgraph_repr
    
    def update_encoding(self, current_encoding, subgraph_repr, hidden_state=None):
        """
        Update the encoding using LSTM with the new subgraph representation
        
        Args:
            current_encoding: Current encoding [1, 1024]
            subgraph_repr: New subgraph representation [1, hidden_dim]
            hidden_state: Optional tuple of (hidden_state, cell_state) from previous step
            
        Returns:
            Tuple of (updated_encoding, new_hidden_state)
        """
        # Initialize hidden state if not provided
        if hidden_state is None:
            h0 = self.init_lstm_hidden.expand(1, 1, self.lstm_hidden_dim).contiguous()
            c0 = self.init_lstm_cell.expand(1, 1, self.lstm_hidden_dim).contiguous()
        else:
            h0, c0 = hidden_state
        
        # Process through LSTM
        _, (hn, cn) = self.lstm(subgraph_repr.unsqueeze(0), (h0, c0))
        
        # Project back to original dimension
        updated_encoding = self.projection(hn.squeeze(0))
        
        # Combine with original encoding using residual connection
        updated_encoding = updated_encoding + current_encoding
        
        return updated_encoding, (hn, cn)

def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                  edge_weight,
                                  size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError

class GLASSConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.gat = GATConv(in_channels, out_channels, heads=1)
        #self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_attr, mask):
        
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        #print(x1.shape, x0.shape)
        #print(mask.shape)
        # mix transformed feature.
        mask = mask.unsqueeze(-1)
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0, self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        x = self.gat(x, edge_index, edge_attr)
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x

class RetrievalPolicy(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_heads=4, lstm_hidden_dim=512, z_ratio=0.8, dropout=0.2, num_layers=2):
        super().__init__()
    
        # Add normalization layers
        self.subgraph_head = nn.Linear(1024, hidden_dim)
        
        self.node_subgraph_mix = nn.Linear(node_dim + hidden_dim, node_dim)
        self.input_norm = nn.LayerNorm(node_dim, eps=1e-4)

        self.question_input = nn.Linear(node_dim, hidden_dim) 
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(node_dim, hidden_dim)

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()
        self.node_question_mix = nn.ModuleList()
        self.edge_question_mix = nn.ModuleList()

        for _ in range(num_layers):

            self.node_question_mix.append(nn.Linear(hidden_dim + hidden_dim, hidden_dim))
            self.edge_question_mix.append(nn.Linear(hidden_dim + hidden_dim, hidden_dim))
            #can we feed in a list of questions, get a list of embeddings then average or something?

            self.convs.append(
                GLASSConv(in_channels=hidden_dim,
                     out_channels=hidden_dim,
                     activation=nn.ReLU()))
            
            self.gns.append(GraphNorm(hidden_dim))
        
        self.policy_head = nn.Linear(hidden_dim, 1)  # Policy output
        
        # Add value function head for baseline
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None, action_mask=None, action_bias=None):

        x = F.relu(self.node_input(x_)) #num_nodes x hidden_dim
        e = F.relu(self.edge_input(edge_attr)) #num_edges x hidden_dim
        q = F.relu(self.question_input(question_embeddings)).squeeze(0) #num_questions x hidden_dim

        num_questions = q.size(0)
        num_nodes = x.size(0)
        num_edges = e.size(0)

        for layer in range(self.num_layers):
            
            q_x = q.expand(num_nodes, -1) #[num_questions, num_nodes, hidden_dim]
            x_combined = torch.cat([x, q_x], dim=1)  # [num_questions, num_nodes, 2*hidden_dim]
            x_combined = F.relu(self.node_question_mix[layer](x_combined)) #[num_questions, num_nodes, hidden_dim]

            q_edge = q.expand(num_edges, -1) #[num_questions, num_edges, hidden_dim]
            e_combined = torch.cat([e, q_edge], dim=1) #[num_questions, num_edges, 2*hidden_dim]
            e_combined = F.relu(self.edge_question_mix[layer](e_combined)) #[num_questions, num_edges, hidden_dim]

            x = self.convs[layer](x_combined, edge_index, e_combined, subgraph_mask) #[num_questions, num_nodes, hidden_dim]  
            x = self.gns[layer](x)

        logits = self.policy_head(x).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        if action_bias is not None:
            logits = logits + torch.log(action_bias + 1e-10)
        
        # Action probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy for regularization (to be used in training loop)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Compute state value for baseline
        state_value = self.value_head(x).mean()
        
        return probs, state_value, x, entropy
        