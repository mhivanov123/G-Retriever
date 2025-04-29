import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
#from torch_geometric.data import Data, Batch
#from torch_geometric.utils import subgraph

from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool
from torch.utils.checkpoint import checkpoint

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
                 dropout=0):
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

class SubgraphRepresentation(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_layers=2):
        super().__init__()

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

            self.convs.append(
                GLASSConv(in_channels=hidden_dim,
                     out_channels=hidden_dim,
                     activation=nn.ReLU()))
            
            self.gns.append(GraphNorm(hidden_dim))

        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool
        self.add_pool = global_add_pool

        self.final_projection = nn.Linear(hidden_dim*3, hidden_dim)

    
    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None):

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

        if subgraph_mask is not None:

            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
            subgraph_x = x[subgraph_mask]
            subgraph_batch = torch.zeros(subgraph_x.size(0), dtype=torch.long, device=x.device)

            x_mean = self.mean_pool(subgraph_x, subgraph_batch) 
            x_max = self.max_pool(subgraph_x, subgraph_batch)
            x_add = self.add_pool(subgraph_x, subgraph_batch)

        else:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
            x_mean = self.mean_pool(x, batch)
            x_max = self.max_pool(x, batch)
            x_add = self.add_pool(x, batch)

        x_combined = torch.cat([x_mean, x_max, x_add], dim=1)
        x = self.final_projection(x_combined)

        return x

class RetrievalPolicy(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, z_ratio=0.8, dropout=0.2, num_layers=2):
        super().__init__()
    
        # Add normalization layers
        self.subgraph_head = nn.Linear(1024, hidden_dim)
        
        self.node_subgraph_mix = nn.Linear(node_dim + hidden_dim, node_dim)
        self.input_norm = nn.LayerNorm(node_dim, eps=1e-4)

        self.question_input = nn.Linear(node_dim, hidden_dim) #TEMPORARY FIXX
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
        
class RetrievalPolicyTriple(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_heads=4, lstm_hidden_dim=512, z_ratio=0.8, dropout=0.2, num_layers=2):
        super().__init__()
    
        # Add normalization layers
        self.subgraph_head = nn.Linear(1024, hidden_dim)
        
        self.node_subgraph_mix = nn.Linear(node_dim + hidden_dim, node_dim)
        self.input_norm = nn.LayerNorm(node_dim, eps=1e-4)

        self.question_input = nn.Linear(node_dim//3, hidden_dim) #TEMPORARY FIXX
        self.node_input = nn.Linear(node_dim, hidden_dim)

        self.num_layers = num_layers

        self.convs_subgraph = nn.ModuleList()
        self.convs_action = nn.ModuleList()
        
        self.gns_subgraph = nn.ModuleList()
        self.gns_action = nn.ModuleList()

        self.subgraph_action_mix = nn.ModuleList()
        self.node_question_mix = nn.ModuleList()

        self.node_question_mix = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        for _ in range(num_layers):

            self.convs.append(
                GLASSConv(in_channels=hidden_dim,
                     out_channels=hidden_dim,
                     activation=nn.ReLU()))
            
            self.gns.append(GraphNorm(hidden_dim))

            #self.subgraph_action_mix.append(nn.Linear(hidden_dim + hidden_dim, hidden_dim))
            #self.subgraph_action_mix.append(nn.Linear(hidden_dim, hidden_dim))

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Add value function head for baseline
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_, edge_index, question_embeddings, subgraph_mask = None, action_mask=None, action_bias=None):

        x = F.relu(self.node_input(x_)) #num_nodes x hidden_dim
        q = F.relu(self.question_input(question_embeddings)).squeeze(0) #num_questions x hidden_dim

        num_nodes = x.size(0)

        q_x = q.expand(num_nodes, -1) #[num_questions, num_nodes, hidden_dim]
        x = torch.cat([x, q_x], dim=1)  # [num_questions, num_nodes, 2*hidden_dim]
        x = F.relu(self.node_question_mix(x)) #[num_questions, num_nodes, hidden_dim]

        for layer in range(self.num_layers):

            x = self.convs[layer](x, edge_index, None, subgraph_mask) #[num_questions, num_nodes, hidden_dim]  
            x = self.gns[layer](x)

            #x = F.relu(self.subgraph_action_mix[layer](x))

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
        state_value = self.value_head(x)[action_mask].mean()
        
        return probs, state_value, x, entropy

class RetrievalPolicyTriple2(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_layers=2):
        super().__init__()
    
        self.question_input = nn.Linear(node_dim, hidden_dim) #TEMPORARY FIXX
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(node_dim, hidden_dim)

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()
        self.subgraph_action_mix = nn.ModuleList()
        
        #self.node_question_mix = nn.ModuleList()

        self.node_question_mix = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        for _ in range(num_layers):

            self.gns.append(GraphNorm(hidden_dim))
            #self.subgraph_action_mix.append(nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(
                GLASSConv(in_channels=hidden_dim,
                     out_channels=hidden_dim,
                     activation=nn.ReLU()))
            
        self.triple_mix = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Add value function head for baseline
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # self.value_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid()
        # )

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None, action_mask=None, action_bias=None):
        
        x = F.relu(self.node_input(x_)) #num_nodes x hidden_dim
        q = F.relu(self.question_input(question_embeddings)).squeeze(0) #num_questions x hidden_dim
        e = F.relu(self.edge_input(edge_attr)) #num_edges x hidden_dim
        num_nodes = x.size(0)

        q_x = q.expand(num_nodes, -1) #[num_questions, num_nodes, hidden_dim]
        x = torch.cat([x, q_x], dim=1)  # [num_questions, num_nodes, 2*hidden_dim]
        x = F.relu(self.node_question_mix(x)) #[num_questions, num_nodes, hidden_dim]

        for layer in range(self.num_layers):

            x = self.convs[layer](x, edge_index, e, subgraph_mask) #[num_questions, num_nodes, hidden_dim]  
            x = self.gns[layer](x)
            #x = F.relu(self.subgraph_action_mix[layer](x))

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        # 1a. Create features for the regular "triple" nodes
        src_node_features = x[src_nodes]
        dst_node_features = x[dst_nodes]

        triple_features = torch.cat([src_node_features, e, dst_node_features], dim=1)
        triple_features = self.triple_mix(triple_features)

        logits = self.policy_head(triple_features).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        if action_bias is not None:
            logits = logits + torch.log(action_bias + 1e-10)
            #logits = logits + action_bias
        # Action probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy for regularization (to be used in training loop)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        action_values = self.value_head(triple_features).squeeze(-1)
        action_values = action_values.masked_fill(~action_mask, 0)
        state_value = (action_values * probs).sum()

        return probs, state_value, triple_features, entropy
    
class SL_warmup_policy(nn.Module):
    def __init__(self, policy, hidden_dim=256):
        super().__init__()

        self.RL_policy = policy
        self.hidden_dim = hidden_dim

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None, action_mask=None):
        
        _, _, x, _ = self.RL_policy(x_, edge_index, edge_attr, question_embeddings, subgraph_mask, action_mask)

        #return F.sigmoid(self.classification_head(x))
        return self.classification_head(x)
        
class customGCN(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_layers=2):
        super().__init__()
    
        self.question_input = nn.Linear(node_dim, hidden_dim) #TEMPORARY FIXX
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Linear(node_dim, hidden_dim)

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()
        
        #self.node_question_mix = nn.ModuleList()

        self.node_question_mix = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        for _ in range(num_layers):

            self.gns.append(GraphNorm(hidden_dim))
            #self.subgraph_action_mix.append(nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(
                GLASSConv(in_channels=hidden_dim,
                     out_channels=hidden_dim,
                     activation=nn.ReLU()))

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None):
        
        x = F.relu(self.node_input(x_)) #num_nodes x hidden_dim
        q = F.relu(self.question_input(question_embeddings)).squeeze(0) #num_questions x hidden_dim
        e = F.relu(self.edge_input(edge_attr)) #num_edges x hidden_dim
        num_nodes = x.size(0)

        q_x = q.expand(num_nodes, -1) #[num_questions, num_nodes, hidden_dim]
        x = torch.cat([x, q_x], dim=1)  # [num_questions, num_nodes, 2*hidden_dim]
        x = F.relu(self.node_question_mix(x)) #[num_questions, num_nodes, hidden_dim]

        for layer in range(self.num_layers):

            x = self.convs[layer](x, edge_index, e, subgraph_mask) #[num_questions, num_nodes, hidden_dim]  
            x = self.gns[layer](x)

        return x, edge_index, e

class SL_classifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=256):
        super().__init__()

        self.custom_GCN = customGCN(node_dim, hidden_dim)
        
        self.classification_head = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x_, edge_index, edge_attr, question_embeddings, question_mask = None):

        x, edge_index, e = self.custom_GCN(x_, edge_index, edge_attr, question_embeddings, question_mask)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        # 1a. Create features for the regular "triple" nodes
        src_node_features = x[src_nodes]
        dst_node_features = x[dst_nodes]

        triple_features = torch.cat([src_node_features, e, dst_node_features], dim=1)

        logits = self.classification_head(triple_features).squeeze(-1)

        return logits, triple_features, x, edge_index, e
    
class Policy(nn.Module):
    def __init__(self, pretrained_classifier, node_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.pretrained_classifier = pretrained_classifier
        self.custom_GCN = customGCN(node_dim, node_dim, num_layers)
        self.critic = Critic(hidden_dim)

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None, question_mask=None, action_mask=None, action_bias=None, use_checkpoint=True):
        if use_checkpoint:
            # Use checkpointing for training
            x, edge_index, e = checkpoint(self.custom_GCN, x_, edge_index, edge_attr, question_embeddings, subgraph_mask)
        else:
            # Skip checkpointing for inference
            x, edge_index, e = self.custom_GCN(x_, edge_index, edge_attr, question_embeddings, subgraph_mask)
        
        # Regular forward pass through pretrained classifier
        if use_checkpoint:
            logits, triple_features = checkpoint(self.pretrained_classifier,
            x, edge_index, e, question_embeddings, question_mask
        )
        else:
            logits, triple_features = self.pretrained_classifier(
                x, edge_index, e, question_embeddings, question_mask
            )
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        if action_bias is not None:
            logits = logits + torch.log(action_bias + 1e-10)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        state_value = self.critic(triple_features, action_mask)

        return probs, state_value, triple_features, entropy

class Policy_test(nn.Module):
    def __init__(self, pretrained_classifier, node_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.pretrained_classifier = pretrained_classifier
        self.critic = Critic(hidden_dim)

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None, question_mask=None, action_mask=None, action_bias=None, use_checkpoint=True):
        
        # Regular forward pass through pretrained classifier
        if use_checkpoint:
            logits, triple_features, _, _, _ = checkpoint(self.pretrained_classifier,
            x_, edge_index, edge_attr, question_embeddings, question_mask
        )
        else:
            logits, triple_features, _, _, _ = self.pretrained_classifier(
                x_, edge_index, edge_attr, question_embeddings, question_mask
            )
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        if action_bias is not None:
            logits = logits + torch.log(action_bias + 1e-10)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        state_value = self.critic(triple_features, action_mask)

        return probs, state_value, triple_features, entropy

class Critic(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, triple_features, action_mask):
        # triple_features: (num_actions, 3*hidden_dim)
        # action_mask: (num_actions,) - boolean mask

        a = self.att(triple_features).squeeze(-1) # a: (num_actions,)
        a = a.masked_fill(~action_mask, float('-inf')) # a: (num_actions,)

        w = torch.softmax(a, dim=-1).unsqueeze(-1) # w: (num_actions, 1)
        pooled = (w * triple_features).sum(dim=-2) # pooled: (3*hidden_dim,)

        feat = self.value_head(pooled) # feat: (1,)

        return feat.squeeze(-1)
        # returns: scalar (float)

class Policy_freeze_then_train(nn.Module):
    def __init__(self, pretrained_classifier, node_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.pretrained_classifier = pretrained_classifier
        self.question_input = nn.Linear(node_dim, hidden_dim)
        self.custom_GCN = customGCN(hidden_dim, hidden_dim, num_layers)
        self.critic = Critic(hidden_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_, edge_index, edge_attr, question_embeddings, subgraph_mask = None, question_mask=None, action_mask=None, action_bias=None, use_checkpoint=True):
        
        if use_checkpoint:
            _, _, x, edge_index, e = checkpoint(self.pretrained_classifier, x_, edge_index, edge_attr, question_embeddings, question_mask)
        else:
            _, _, x, edge_index, e = self.pretrained_classifier(x_, edge_index, edge_attr, question_embeddings, question_mask)

        q = F.relu(self.question_input(question_embeddings)).squeeze(0) #num_questions x hidden_dim
        
        if use_checkpoint:
            x, edge_index, e = checkpoint(self.custom_GCN, x, edge_index, e, q, subgraph_mask)
        else:
            x, edge_index, e = self.custom_GCN(x, edge_index, e, q, subgraph_mask)
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        # 1a. Create features for the regular "triple" nodes
        src_node_features = x[src_nodes]
        dst_node_features = x[dst_nodes]

        triple_features = torch.cat([src_node_features, e, dst_node_features], dim=1)

        logits = self.policy_head(triple_features).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        if action_bias is not None:
            logits = logits + torch.log(action_bias + 1e-10)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        state_value = self.critic(triple_features, action_mask)

        return probs, state_value, triple_features, entropy
