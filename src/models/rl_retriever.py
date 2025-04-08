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
    def __init__(self, sample, device, policy_net=None, directed=False, triple_graph=False):
        """
        Args:
            sample: Dictionary containing graph and other data
            device: torch device
            policy_net: Policy network with GNN-LSTM for updating encodings
        """

        self.graph = sample['graph']
        self.graph.to(device)
        
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

        self.policy_net = policy_net
        self.question = sample['question']
        
        self.subgraph_encoding = self.q_emb.reshape(1, 1, -1)

        self.triple_graph = triple_graph

        if not all(node in self.shortest_path_nodes for node in self.a_nodes):
            print(f"Answer nodes not in shortest path nodes")
    
    def step(self, action):
        """Execute action and return reward and done flag"""
        action_node = action.item()

        if self.triple_graph:
            action_node = self.graph.edge_index[1, action_node].item()

        if action_node in self.visited_nodes:
            print(f"Invalid action: {action_node}")
        # Regular action processing
        edge_idx = set()
        if self.visited_nodes:
            for current_node in self.visited_nodes:
                edge_mask = ((self.graph.edge_index[0] == current_node) & (self.graph.edge_index[1] == action_node)) | ((self.graph.edge_index[1] == current_node) & (self.graph.edge_index[0] == action_node))
                if edge_mask.any():
                    edge_idx.add(edge_mask.nonzero()[0].item())

        # Update state
        if edge_idx:
            self.visited_edges.update(edge_idx)
        else:
            print(f"Invalid action: {action_node}")
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
    
    def get_valid_actions(self):
        """Returns a mask of valid actions (unvisited edge-node pairs + finish action)"""

        if self.triple_graph:
            return self.get_valid_actions_triple()

        valid_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
            
        for current_node in self.visited_nodes:
            # Check outgoing edges (current_node -> next_node)
            outgoing_edges = (self.graph.edge_index[0] == current_node).nonzero().squeeze(-1)
            
            for edge_idx in outgoing_edges:
                next_node = self.graph.edge_index[1, edge_idx].item()
                if next_node not in self.visited_nodes and edge_idx.item() not in self.visited_edges:
                    valid_mask[next_node] = True
        
        return valid_mask

    def optimal_valid_actions(self):
        if self.triple_graph:
            return self.optimal_valid_actions_triple()

        legal_actions = self.get_valid_actions()
        valid_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        valid_mask[self.shortest_path_nodes] = True
        return legal_actions & valid_mask
    
    def get_valid_actions_triple(self):
        """Returns a mask of valid actions (unvisited edge-node pairs)"""
        valid_mask = torch.zeros(self.graph.num_edges, dtype=torch.bool, device=self.device)
            
        # Find all edges connected to any frontier node
        for current_node in self.visited_nodes:
            # Check outgoing edges (current_node -> next_node)
            outgoing_edges = (self.graph.edge_index[0] == current_node).nonzero().squeeze(-1)

            for edge_idx in outgoing_edges:
                if self.graph.edge_index[1, edge_idx].item() not in self.visited_nodes:
                    valid_mask[edge_idx] = True
        
        return valid_mask
    
    def optimal_valid_actions_triple(self):

        valid_mask = self.get_valid_actions_triple()
        # Create a mask for edges that lead to nodes in the shortest path
        target_nodes = torch.tensor(list(self.shortest_path_nodes), device=self.device)
        edge_targets = self.graph.edge_index[1]
        optimal_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        
        for node in target_nodes:
            optimal_mask = optimal_mask | (edge_targets == node)

        return valid_mask & optimal_mask
    
    def get_subgraph_mask(self):
        """Returns a mask of the current subgraph"""
        mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        mask[list(self.visited_nodes)] = True
        return mask

    def get_biased_action_mask(self, teacher_bias=1.0):
        """Returns a mask with higher values for teacher-recommended actions"""
        # Get regular valid actions (boolean mask)
        valid_mask = self.get_valid_actions()
        
        # Convert to float mask with 1.0 for valid actions
        action_bias = valid_mask.float()

        optimal_mask = self.optimal_valid_actions()
        action_bias[optimal_mask] = teacher_bias

        return action_bias, valid_mask
    
    def get_subgraph(self):
        """Returns the constructed subgraph as edge and node sets"""
        return self.visited_nodes, self.visited_edges

class RetrievalTrainer:
    def __init__(self, policy_net, ppo_epochs=4, tf_start_bias=10.0, 
                 tf_end_bias=1.0, tf_total_epochs=100, directed=False, triple_graph=False,
                 ppo_batch_size=8):
        """
        Args:
            policy_net: Policy network
            ppo_epochs: Number of PPO update epochs
            tf_start_bias: Initial teacher forcing bias strength
            tf_end_bias: Final teacher forcing bias strength
            tf_total_epochs: Total epochs over which to anneal teacher forcing
            ppo_batch_size: Number of trajectories to collect before performing a PPO update
        """
        self.policy_net = policy_net
        self.ppo_epochs = ppo_epochs
        
        # Teacher forcing parameters
        self.use_teacher_forcing = True
        self.tf_start_bias = tf_start_bias
        self.tf_end_bias = tf_end_bias
        self.tf_total_epochs = tf_total_epochs
        self.current_epoch = 0
        self.directed = directed
        self.triple_graph = triple_graph
        self.ppo_batch_size = ppo_batch_size
        
        # Simple trajectory buffer
        self.trajectories = []
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    
    def set_teacher_forcing(self, use_teacher_forcing):
        """Set whether to use teacher forcing"""

        self.use_teacher_forcing = use_teacher_forcing
    
    def set_current_epoch(self, epoch):
        """Update the current epoch for annealing calculations"""
        self.current_epoch = epoch
    
    def get_current_teacher_bias(self):
        """Calculate the current teacher bias based on annealing schedule"""
        if not self.use_teacher_forcing:
            return 1.0  # No bias when teacher forcing is disabled
        
        # Linear annealing from start_bias to end_bias
        progress = min(1.0, self.current_epoch / self.tf_total_epochs)
        current_bias = self.tf_start_bias - progress * (self.tf_start_bias - self.tf_end_bias)
        return current_bias
    
    def train_step(self, sample, max_steps=100):
        """
        Collect a single trajectory and add to buffer.
        If buffer reaches batch_size, perform PPO update.
        """
        # Collect trajectory
        trajectory, stats = self._collect_trajectory(sample, max_steps)
        
        # Add to buffer
        self.trajectories.append(trajectory)
        
        # If we have enough trajectories, perform batch update
        if len(self.trajectories) >= self.ppo_batch_size:
            # Perform PPO update and get loss statistics
            loss_stats = self._batch_update()
            
            # Clear buffer after update
            self.trajectories = []
            
            # Add loss statistics to return dict
            stats.update(loss_stats)
        else:
            # Add placeholder loss values when no update is performed
            stats.update({
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0
            })
            
        return stats
    
    def _collect_trajectory(self, sample, max_steps=100):
        """Collect a single trajectory and return it with statistics"""
        self.policy_net.train()
        
        # Initialize containers for trajectory
        states = []
        actions = []
        action_log_probs = []
        values = []
        rewards = []
        masks = []
        
        # Initialize state
        device = next(self.policy_net.parameters()).device
        state = RetrievalState(sample, device, policy_net=self.policy_net, 
                              directed=self.directed, triple_graph=self.triple_graph)
        
        # Get teacher bias
        current_teacher_bias = self.get_current_teacher_bias()
        
        # Collect trajectory
        for step in range(max_steps):
            # Get state representation
            subgraph_mask = state.get_subgraph_mask()
            action_bias, valid_mask = state.get_biased_action_mask(teacher_bias=current_teacher_bias)

            if not valid_mask.any():
                print("No valid actions")
                break
            
            if all([node in state.visited_nodes for node in state.shortest_path_nodes]):
                print("All shortest path nodes visited")
                break
            
            # Store current state
            states.append({
                'x': state.graph.x.clone().detach(),
                'edge_index': state.graph.edge_index.clone().detach(),
                'edge_attr': state.graph.edge_attr.clone().detach(),
                'question_embeddings': state.subgraph_encoding.clone().detach(),
                'subgraph_mask': subgraph_mask.clone(),
                'action_mask': valid_mask.clone(),
                'action_bias': action_bias.clone()
            })

            # Get action probabilities
            node_probs, state_value, _, entropy = self.policy_net(
                x_ = state.graph.x.float(), 
                edge_index = state.graph.edge_index.long(),
                edge_attr = state.graph.edge_attr.float(),
                question_embeddings = state.subgraph_encoding,
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
            
            # Execute action
            reward, done = state.step(action)
            rewards.append(reward)
            masks.append(1.0 - float(done))
            
            if done:
                break
        
        # Create trajectory dictionary
        trajectory = {
            'states': states,
            'actions': actions,
            'action_log_probs': action_log_probs,
            'values': values,
            'rewards': rewards,
            'masks': masks
        }
        
        # Calculate statistics for this trajectory
        stats = {
            'reward': sum(rewards),
            'steps': len(rewards),
            'visited_nodes': len(state.visited_nodes),
            'visited_edges': len(state.visited_edges),
            'perc_correct_nodes': sum(node in state.shortest_path_nodes for node in state.visited_nodes)/max(1, len(state.shortest_path_nodes)),
            'perc_answer_nodes_reached': sum(node in state.a_nodes for node in state.visited_nodes)/max(1, len(state.a_nodes))
        }
        
        return trajectory, stats
    
    def _batch_update(self):
        """Perform PPO update on all trajectories in the buffer"""
        device = next(self.policy_net.parameters()).device
        
        # Initialize loss accumulators
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        # First, compute returns and advantages for all trajectories
        all_advantages = []
        
        # Process each trajectory to compute returns and advantages
        for trajectory in self.trajectories:
            # Compute returns and advantages for this trajectory
            returns, advantages = self._compute_returns_and_advantages(trajectory, device)
            
            # Add computed values to the trajectory
            trajectory['returns'] = returns
            trajectory['advantages'] = advantages
            
            # Collect all advantages for normalization
            all_advantages.extend(advantages)
        
        # Normalize advantages across all trajectories
        if len(all_advantages) > 1:
            all_advantages_tensor = torch.tensor(all_advantages, device=device)
            mean_adv = all_advantages_tensor.mean()
            std_adv = all_advantages_tensor.std() + 1e-8
            
            # Normalize advantages in each trajectory
            for trajectory in self.trajectories:
                trajectory['advantages'] = [(adv - mean_adv) / std_adv for adv in trajectory['advantages']]
        
        # Perform multiple epochs of PPO updates
        for _ in range(self.ppo_epochs):
            # Collect all data from trajectories for batch processing
            batch_states = []
            batch_actions = []
            batch_old_log_probs = []
            batch_returns = []
            batch_advantages = []
            
            # Gather data from all trajectories
            for trajectory in self.trajectories:
                batch_states.extend(trajectory['states'])
                batch_actions.extend(trajectory['actions'])
                batch_old_log_probs.extend(trajectory['action_log_probs'])
                batch_returns.extend(trajectory['returns'])
                batch_advantages.extend(trajectory['advantages'])
            
            # Convert to tensors where appropriate
            batch_actions = torch.stack(batch_actions)
            batch_old_log_probs = torch.stack(batch_old_log_probs)
            batch_returns = torch.tensor(batch_returns, device=device)
            batch_advantages = torch.tensor(batch_advantages, device=device)
            
            # Process in mini-batches to avoid memory issues
            mini_batch_size = 64
            indices = torch.randperm(len(batch_states))
            
            for start_idx in range(0, len(batch_states), mini_batch_size):
                # Get mini-batch indices
                mb_indices = indices[start_idx:start_idx + mini_batch_size]
                
                # Initialize loss components for this mini-batch
                mb_policy_loss = 0
                mb_value_loss = 0
                mb_entropy_loss = 0
                
                # Zero gradients once for the mini-batch
                self.optimizer.zero_grad()
                
                # Process each state in the mini-batch
                for i in mb_indices:
                    state_info = batch_states[i]
                    
                    # Forward pass through policy network
                    action_probs, state_value, _, entropy = self.policy_net(
                        state_info['x'].float(),
                        state_info['edge_index'].long(),
                        state_info['edge_attr'].float(),
                        state_info['question_embeddings'],
                        subgraph_mask=state_info['subgraph_mask'],
                        action_mask=state_info['action_mask'],
                        action_bias=state_info['action_bias']
                    )
                    
                    # Get log probability of the action that was taken
                    log_prob = torch.log(action_probs + 1e-10).gather(0, batch_actions[i].unsqueeze(0)).squeeze(0)
                    
                    # Calculate importance sampling ratio
                    ratio = torch.exp(log_prob - batch_old_log_probs[i])
                    
                    # PPO clipped objective
                    clip_param = 0.2
                    surr1 = ratio * batch_advantages[i]
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages[i]
                    policy_loss_i = -torch.min(surr1, surr2)
                    
                    # Value function loss
                    value_loss_i = 0.5 * (state_value - batch_returns[i]).pow(2)
                    
                    # Entropy bonus
                    entropy_loss_i = -entropy
                    
                    # Accumulate losses for this mini-batch
                    mb_policy_loss += policy_loss_i
                    mb_value_loss += value_loss_i
                    mb_entropy_loss += entropy_loss_i
                
                # Average losses for this mini-batch
                mb_size = len(mb_indices)
                if mb_size > 0:
                    mb_policy_loss /= mb_size
                    mb_value_loss /= mb_size
                    mb_entropy_loss /= mb_size
                
                # Compute total loss
                loss = mb_policy_loss + 0.5 * mb_value_loss + 0.01 * mb_entropy_loss
                
                # Backward pass and optimization step
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()
                
                # Accumulate losses for reporting
                total_loss += loss.item()
                policy_loss += mb_policy_loss.item()
                value_loss += mb_value_loss.item()
                entropy_loss += mb_entropy_loss.item()
        
        # Calculate average losses
        num_mini_batches = (len(batch_states) + mini_batch_size - 1) // mini_batch_size
        num_updates = num_mini_batches * self.ppo_epochs
        if num_updates > 0:
            total_loss /= num_updates
            policy_loss /= num_updates
            value_loss /= num_updates
            entropy_loss /= num_updates
        
        return {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy_loss
        }
    
    def _compute_returns_and_advantages(self, trajectory, device, gamma=0.99, gae_lambda=0.95):
        """Compute returns and advantages for a single trajectory"""
        rewards = trajectory['rewards']
        values = trajectory['values']
        masks = trajectory['masks']
        
        returns = []
        advantages = []
        
        # Terminal state has value 0
        next_value = 0
        next_advantage = 0
        
        # Compute GAE and returns in reverse order
        for t in reversed(range(len(rewards))):
            # Compute TD error: r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            
            # Compute advantage using GAE
            advantage = delta + gamma * gae_lambda * next_advantage * masks[t]
            
            # Store advantage and return
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
            
            # Update for next iteration
            next_value = values[t]
            next_advantage = advantage
        
        return returns, advantages
    
    def inference_step(self, sample, max_steps=100):
        """
        Run inference to get a subgraph starting from given nodes
        """
        self.policy_net.eval()
        device = next(self.policy_net.parameters()).device
        
        # Initialize state without supervision
        state = RetrievalState(sample, device, policy_net=self.policy_net, 
                               directed=self.directed, triple_graph=self.triple_graph)
        
        for _ in range(max_steps):
            valid_mask = state.get_valid_actions()
            
            if not valid_mask.any():
                break
                
            # Get action probabilities using the updated model
            #if not self.triple_graph:
            probs, _, _, _ = self.policy_net(
                x_ = state.graph.x.float(), 
                edge_index = state.graph.edge_index.long(),
                edge_attr = state.graph.edge_attr.float(),
                question_embeddings = state.subgraph_encoding,
                action_mask=valid_mask,
                subgraph_mask=state.get_subgraph_mask()
            )
            '''else:
                probs, _, _, _ = self.policy_net(
                    state.graph.x.float(), 
                    state.graph.edge_index.long(),
                    state.subgraph_encoding,
                    action_mask=valid_mask,
                    subgraph_mask=state.get_subgraph_mask()
                )'''
            
            # Take most probable valid action
            action = probs.argmax()
            
            # Execute action
            reward, done = state.step(action)
            
            if done:
                break
                
        return state.get_subgraph()

        