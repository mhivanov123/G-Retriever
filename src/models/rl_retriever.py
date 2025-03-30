import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class RetrievalState:
    def __init__(self, sample, device, encoder=None, policy_net=None):
        """
        Args:
            sample: Dictionary containing graph and other data
            device: torch device
            encoder: Optional text encoder for initial encoding
            policy_net: Policy network with GNN-LSTM for updating encodings
        """

        self.graph = sample['graph'].to(device)
        self.q_nodes = sample['q_idx']
        self.a_nodes = set(sample['a_idx'])
        self.shortest_path_nodes = sample['shortest_path_nodes'] if 'shortest_path_nodes' in sample else None
        self.node_dict = sample['id_to_node']
        self.edge_dict = sample['edge_id_to_row']
        self.device = device

        self.q_emb = sample['q_emb'].float().to(self.device)
        self.rewritten_question = sample['rewritten_question']
        self.rewritten_question_emb = sample['rewritten_question_emb'].float().to(self.device)
        
        self.visited_nodes = set(self.q_nodes)
        self.visited_edges = set()

        self.encoder = encoder
        self.policy_net = policy_net
        self.question = sample['question']
        
        self.subgraph_encoding = self.q_emb.reshape(1, 1, -1)
    
    def get_valid_actions(self):
        """Returns a mask of valid actions (unvisited edge-node pairs + finish action)"""
        valid_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
            
        # Find all edges connected to any frontier node
        for current_node in self.visited_nodes:
            # Check outgoing edges (current_node -> next_node)
            outgoing_edges = (self.graph.edge_index[0] == current_node).nonzero().squeeze(-1)
            
            for edge_idx in outgoing_edges:
                next_node = self.graph.edge_index[1, edge_idx].item()
                if next_node not in self.visited_nodes and edge_idx.item() not in self.visited_edges:
                    valid_mask[next_node] = True
            
            # Check incoming edges (next_node -> current_node)
            incoming_edges = (self.graph.edge_index[1] == current_node).nonzero().squeeze(-1)
            
            for edge_idx in incoming_edges:
                prev_node = self.graph.edge_index[0, edge_idx].item()
                if prev_node not in self.visited_nodes and edge_idx.item() not in self.visited_edges:
                    valid_mask[prev_node] = True
        
        return valid_mask

    def optimal_valid_actions(self):
        """Returns a mask of valid actions (unvisited edge-node pairs + finish action)"""
        legal_actions = self.get_valid_actions()
        valid_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        valid_mask[self.shortest_path_nodes] = True
        return legal_actions & valid_mask
    
    def get_subgraph_mask(self):
        """Returns a mask of the current subgraph"""
        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        mask[list(self.visited_nodes)] = True
        return mask
    
    def step(self, action):
        """Execute action and return reward and done flag"""
        action_node = action.item()
            
        # Regular action processing
        edge_idx = set()
        if self.visited_nodes:
            for current_node in self.visited_nodes:
                edge_mask = (self.graph.edge_index[0] == current_node) & (self.graph.edge_index[1] == action_node)
                if edge_mask.any():
                    edge_idx.add(edge_mask.nonzero()[0].item())

        # Update state
        if edge_idx:
            self.visited_edges.update(edge_idx)
        self.visited_nodes.add(action_node)
        
        # Compute reward for the action
        if self.shortest_path_nodes is not None:
            if action_node in self.a_nodes:
                reward = 10.0  # Keep high reward for finding answer
            elif action_node in self.shortest_path_nodes:
                # Scale intermediate rewards based on path progress
                progress = len(self.visited_nodes.intersection(self.shortest_path_nodes)) / len(self.shortest_path_nodes)
                reward = 1.0 + 4.0 * progress  # Gradually increase rewards as we make progress
            else:
                reward = -0.1
            
        return reward, False
    
    def get_subgraph(self):
        """Returns the constructed subgraph as edge and node sets"""
        return self.visited_nodes, self.visited_edges

    def get_biased_action_mask(self, teacher_bias=5.0):
        """Returns a mask with higher values for teacher-recommended actions"""
        # Get regular valid actions (boolean mask)
        valid_mask = self.get_valid_actions()
        
        # Convert to float mask with 1.0 for valid actions
        action_bias = valid_mask.float()

        optimal_mask = self.optimal_valid_actions()
        action_bias[optimal_mask] = teacher_bias

        return action_bias, valid_mask

class RetrievalTrainer:
    def __init__(self, policy_net, encoder, lr=1e-4, gamma=0.99, max_grad_norm=1.0, 
                 value_loss_coef=0.5, entropy_coef=0.01, ppo_epochs=4, 
                 clip_param=0.2, target_kl=0.01, teacher_forcing = False):
        
        self.policy_net = policy_net
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.teacher_forcing = teacher_forcing
        
        # PPO specific parameters
        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        self.target_kl = target_kl

    def set_teacher_forcing(self, teacher_forcing):
        self.teacher_forcing = teacher_forcing

    def train_step(self, sample, max_steps=100):
        """
        Single training episode using Actor-Critic with entropy regularization
        """
        # Skip samples with no question or answer nodes
        if not sample['q_idx'] or not sample['a_idx'] or not sample['shortest_path_nodes']:
            return {
                'loss': 0.0,
                'reward': 0.0,
                'steps': 0,
                'visited_nodes': 0,
                'visited_edges': 0,
                'perc_correct_nodes': 0.0,
                'perc_answer_nodes_reached': 0.0
            }
            
        torch.autograd.set_detect_anomaly(True)

        self.policy_net.train()
        
        # Initialize episode storage
        states = []
        actions = []
        action_log_probs = []
        values = []
        rewards = []
        masks = []  # For tracking episode termination
        entropies = []

        # Convert graph to device and initialize state
        device = next(self.policy_net.parameters()).device
        state = RetrievalState(sample, device, encoder=self.encoder, policy_net=self.policy_net)
        
        # Calculate teacher forcing strength based on current probability
        teacher_bias = 1.0 + 9.0 * self.teacher_forcing  # Scales from 10.0 to 1.0
        
        # Collect trajectory
        for step in range(max_steps):
            # Get action biases and valid mask
            action_bias, valid_mask = state.get_biased_action_mask(teacher_bias=teacher_bias)
            subgraph_mask = state.get_subgraph_mask()
            
            # Check if there are any valid actions
            if not valid_mask.any() or all([node in state.visited_nodes for node in state.shortest_path_nodes]):
                break
                # No valid actions, must terminate
                action = torch.tensor(state.graph.num_nodes, device=device)
                reward, done = state.step(action)
                rewards.append(reward)
                masks.append(0.0)  # Episode terminated
                break
            
            # Store current state info
            states.append({
                'x': state.graph.x.clone().detach(),
                'edge_index': state.graph.edge_index.clone().detach(),
                'edge_attr': state.graph.edge_attr.clone().detach(),
                'question_embeddings': state.subgraph_encoding.clone().detach(),
                'subgraph_mask': subgraph_mask.clone(),
                'action_mask': valid_mask.clone(),
                'action_bias': action_bias.clone()
            })
            
            # Get action probabilities with biasing
            node_probs, state_value, _, entropy = self.policy_net(
                state.graph.x.float(), 
                state.graph.edge_index.long(),
                state.graph.edge_attr.float(),
                state.subgraph_encoding,
                subgraph_mask=subgraph_mask,
                action_mask=valid_mask,
                action_bias=action_bias
            )
            
            # Sample action
            node_dist = torch.distributions.Categorical(node_probs)
            action = node_dist.sample()
            
            # Store trajectory information
            actions.append(action)
            action_log_probs.append(node_dist.log_prob(action).detach())
            values.append(state_value.detach())
            entropies.append(entropy.detach())
            
            # Execute action
            reward, done = state.step(action)
            rewards.append(reward)
            masks.append(1.0 - float(done))  # 0 if done, 1 otherwise
            
            if done:
                break

        trajectory = {
            'states': states,
            'actions': actions,
            'action_log_probs': action_log_probs,
            'values': values,
            'rewards': rewards,
            'masks': masks
        }
        
        total_loss, policy_loss, value_loss, entropy_loss = ppo_update(self.policy_net, self.optimizer, device, trajectory)
        
        # Average losses over updates
        num_updates = len(states) * self.ppo_epochs
        if num_updates > 0:
            total_loss /= num_updates
            policy_loss /= num_updates
            value_loss /= num_updates
            entropy_loss /= num_updates

        print([sample['id_to_node'][node] for node in state.visited_nodes])

        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy_loss,
            'reward': sum(rewards),
            'steps': len(rewards),
            'visited_nodes': len(state.visited_nodes),
            'visited_edges': len(state.visited_edges),
            'perc_correct_nodes': sum(node in state.shortest_path_nodes for node in state.visited_nodes)/max(1, len(state.visited_nodes)),
            'perc_answer_nodes_reached': sum(node in state.a_nodes for node in state.visited_nodes)/max(1, len(state.a_nodes))
        }

    def inference_step(self, sample, max_steps=100):
        """
        Run inference to get a subgraph starting from given nodes
        """
        self.policy_net.eval()
        device = next(self.policy_net.parameters()).device
        
        # Initialize state without supervision
        state = RetrievalState(sample, device, encoder=self.encoder, policy_net=self.policy_net)
        
        for _ in range(max_steps):
            valid_mask = state.get_valid_actions()
            if not valid_mask.any():
                break
                
            # Get action probabilities using the updated model
            probs, _, _, _ = self.policy_net(
                state.graph.x.float(), 
                state.graph.edge_index.long(),
                state.graph.edge_attr.float(),
                state.subgraph_encoding,
                action_mask=valid_mask,
                subgraph_mask=state.get_subgraph_mask()
            )
            
            # Take most probable valid action
            action = probs.argmax()
            
            # Execute action
            reward, done = state.step(action)
            
            if done:
                break
                
        return state.get_subgraph()

def ppo_update(agent, optimizer, device, trajectory, gamma=0.99, 
               ppo_epochs=4, clip_param=0.2, value_loss_coef=0.5, 
               entropy_coef=0.01, max_grad_norm=1.0):
    """
    Perform a PPO update on the given trajectory
    
    Args:
        agent: Policy network
        optimizer: Optimizer for policy network
        device: Torch device
        trajectory: Dictionary containing trajectory information
        gamma: Discount factor
        ppo_epochs: Number of epochs to update on the same trajectory
        clip_param: PPO clipping parameter
        value_loss_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        total_loss, policy_loss, value_loss, entropy_loss
    """ 
    # Extract trajectory components
    states = trajectory['states']
    actions = trajectory['actions']
    old_log_probs = trajectory['action_log_probs']
    old_values = trajectory['values']
    rewards = trajectory['rewards']
    masks = trajectory['masks']  # 0 if done, 1 otherwise

    # Compute returns and advantages using GAE
    returns = []
    advantages = []
    gae = 0
    next_value = 0  # Terminal state value is 0
    
    for step in reversed(range(len(rewards))):
        # Calculate TD error and GAE
        delta = rewards[step] + gamma * next_value * masks[step] - old_values[step]
        gae = delta + gamma * 0.95 * masks[step] * gae  # 0.95 is GAE lambda
        next_value = old_values[step]
        
        # Store returns and advantages
        returns.insert(0, gae + old_values[step])
        advantages.insert(0, gae)
    
    # Convert lists to tensors
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    old_log_probs = torch.stack(old_log_probs)
    actions = torch.stack(actions)
    
    # Normalize advantages (important for stable training)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Initialize loss accumulators
    total_loss = 0
    policy_loss = 0
    value_loss = 0
    entropy_loss = 0
    
    # Perform multiple epochs of updates
    for _ in range(ppo_epochs):
        # Process each state in the trajectory
        for i in range(len(states)):
            state_info = states[i]
            
            # Forward pass through policy network with the same biasing
            action_probs, state_value, _, entropy = agent(
                state_info['x'].float(),
                state_info['edge_index'].long(),
                state_info['edge_attr'].float(),
                state_info['question_embeddings'],
                subgraph_mask=state_info['subgraph_mask'],
                action_mask=state_info['action_mask'],
                action_bias=state_info['action_bias']
            )
            
            # Get log probability of the action that was taken
            log_prob = torch.log(action_probs + 1e-10).gather(0, actions[i].unsqueeze(0)).squeeze(0)
            
            # Calculate importance sampling ratio
            ratio = torch.exp(log_prob - old_log_probs[i])
            
            # PPO clipped objective
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages[i]
            policy_loss_i = -torch.min(surr1, surr2)
            
            # Value function loss with clipping
            value_pred_clipped = old_values[i] + torch.clamp(
                state_value - old_values[i], 
                -clip_param, 
                clip_param
            )
            value_loss1 = (state_value - returns[i]).pow(2)
            value_loss2 = (value_pred_clipped - returns[i]).pow(2)
            value_loss_i = 0.5 * torch.max(value_loss1, value_loss2)
            
            # Entropy bonus for exploration
            entropy_loss_i = -entropy
            
            # Total loss
            loss = policy_loss_i + value_loss_coef * value_loss_i + entropy_coef * entropy_loss_i
            
            # Accumulate losses for reporting
            total_loss += loss.item()
            policy_loss += policy_loss_i.item()
            value_loss += value_loss_i.item()
            entropy_loss += entropy_loss_i.item()
            
            # Perform optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

    # Average losses over updates
    num_updates = len(states) * ppo_epochs
    if num_updates > 0:
        total_loss /= num_updates
        policy_loss /= num_updates
        value_loss /= num_updates
        entropy_loss /= num_updates

    return total_loss, policy_loss, value_loss, entropy_loss

        