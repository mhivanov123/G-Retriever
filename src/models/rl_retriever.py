import torch
from torch.multiprocessing import Pool
from functools import partial
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import random
class RetrievalState:
    def __init__(self, sample, device, policy_net=None, directed=False, triple_graph=False):
        """
        Args:
            sample: Dictionary containing graph and other data
            device: torch device
            policy_net: Policy network with GNN-LSTM for updating encodings
        """

        self.device = device

        self.graph = sample['graph']
        self.graph.to(device)
        
        self.q_nodes = sample['q_idx']
        self.a_nodes = set(sample['a_idx'])
        self.shortest_path_nodes = sample['shortest_path_nodes'] if 'shortest_path_nodes' in sample else None
        self.node_dict = sample['id_to_node']
        self.edge_dict = sample['edge_id_to_row']

        self.q_emb = sample['q_emb'].reshape(1, 1, -1).float().to(self.device)
        
        self.visited_nodes = set(self.q_nodes)
        self.visited_edges = set()

        self.policy_net = policy_net
        self.question = sample['question']
        self.question_mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool, device=self.device)
        self.question_mask[self.q_nodes] = True

        self.triple_graph = triple_graph

        self.edge_indices = []

        if not all(node in self.shortest_path_nodes for node in self.a_nodes):
            print(f"Answer nodes not in shortest path nodes")
    
    def step(self, action):
        """Execute action and return reward and done flag"""
        action_node = action.item()

        if self.triple_graph:
            action_node = self.graph.edge_index[1, action_node].item()
            self.edge_indices.append(action_node)
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
        #action_bias = torch.ones_like(valid_mask)
        optimal_mask = self.optimal_valid_actions()
        action_bias[optimal_mask] = teacher_bias

        return action_bias, valid_mask
    
    def get_subgraph(self):
        """Returns the constructed subgraph as edge and node sets"""
        return self.visited_nodes, self.visited_edges
    
    def get_answer(self):
        return self.visited_nodes, self.edge_indices
    
    
class RetrievalTrainer:
    def __init__(self, policy_net, optimizer = None, ppo_epochs=4, tf_start_bias=10.0, 
                 tf_end_bias=1.0, tf_total_epochs=100, directed=False, triple_graph=False,
                 ppo_batch_size=8, ppo_minibatch_size=64, num_workers=4):
        """
        Args:
            policy_net: Policy network
            ppo_epochs: Number of PPO update epochs
            tf_start_bias: Initial teacher forcing bias strength
            tf_end_bias: Final teacher forcing bias strength
            tf_total_epochs: Total epochs over which to anneal teacher forcing
            ppo_batch_size: Number of trajectories to collect before performing a PPO update
            num_workers: Number of parallel workers for trajectory collection
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
        self.ppo_minibatch_size = ppo_minibatch_size
        self.num_workers = num_workers
        
        # Simple trajectory buffer
        self.trajectories = []
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
    
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
    
    @staticmethod
    def _collect_trajectory(sample, policy_net, device, teacher_bias, max_steps, directed, triple_graph, epsilon=0.0):
        """Worker function for collecting a single trajectory"""
        # Initialize state
        state = RetrievalState(sample, device, policy_net=policy_net, 
                              directed=directed, triple_graph=triple_graph)
        
        if state.shortest_path_nodes is None or len(state.shortest_path_nodes) == 0 or len(state.a_nodes) == 0 or len(state.q_nodes) == 0:
            return None
        
        # Containers for trajectory
        states = []
        actions = []
        action_log_probs = []
        probs = []
        values = []
        rewards = []
        masks = []
        
        
        # Collect trajectory
        for step in range(max_steps):
            # Get state representation
            subgraph_mask = state.get_subgraph_mask()
            action_bias, valid_mask = state.get_biased_action_mask(teacher_bias=teacher_bias)
            
            if not valid_mask.any():
                break
            
            if all([node in state.visited_nodes for node in state.shortest_path_nodes]):
                break
            
            # Store current state
            states.append({
                'x': state.graph.x.clone().detach(),
                'edge_index': state.graph.edge_index.clone().detach(),
                'edge_attr': state.graph.edge_attr.clone().detach(),
                'question_embeddings': state.q_emb.clone().detach(),
                'subgraph_mask': subgraph_mask.clone(),
                'action_mask': valid_mask.clone(),
                'action_bias': action_bias.clone(),
                'question_mask': state.question_mask.clone()
            })
            
            # Get action probabilities
            with torch.no_grad():
                node_probs, state_value, _, entropy = policy_net(
                    x_=state.graph.x.float(), 
                    edge_index=state.graph.edge_index.long(),
                    edge_attr=state.graph.edge_attr.float(),
                    question_embeddings=state.q_emb.clone().detach(),
                    subgraph_mask=subgraph_mask,
                    action_mask=valid_mask,
                    action_bias=action_bias,
                    question_mask=state.question_mask,
                    use_checkpoint=False
                )
            
            # Sample action
            node_dist = torch.distributions.Categorical(node_probs)
            action = node_dist.sample()

            if torch.rand(1) < epsilon:
                action = random.choice(valid_mask.nonzero().squeeze(-1))
            else:
                action = node_dist.sample()
            
            # Store trajectory information
            actions.append(action)
            action_log_probs.append(node_dist.log_prob(action).detach())
            probs.append(node_probs.detach())  # Store the entire probability distribution
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
            'probs': probs,
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
            'perc_answer_nodes_reached': sum(node in state.a_nodes for node in state.visited_nodes)/max(1, len(state.a_nodes)),
            'perc_any_answer_node_reached': sum(node in state.a_nodes for node in state.visited_nodes) > 0
        }
        
        return trajectory, stats
    
    def train_step(self, sample, max_steps=100):
        """
        Collect a single trajectory and add to buffer.
        If buffer reaches batch_size, perform PPO update.
        """
        # Collect trajectory
        trajectory, stats = self._collect_trajectory(sample, policy_net=self.policy_net, device=next(self.policy_net.parameters()).device, teacher_bias=self.get_current_teacher_bias(), max_steps=max_steps, directed=self.directed, triple_graph=self.triple_graph)
        
        # Add to buffer
        self.trajectories.append(trajectory)
        
        # If we have enough trajectories, perform batch update
        if len(self.trajectories) >= self.ppo_batch_size:
            # Perform PPO update and get loss statistics
            loss_stats = self._batch_update()
            #loss_stats = self._batch_update_reinforce()
            
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
                'entropy_loss': 0.0,
                'valid_steps': 0,
                'valid_trajectories': 0
            })
            
        return stats
    
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
        for epoch in range(self.ppo_epochs):
            # Collect all data from trajectories for batch processing
            batch_states = []
            batch_actions = []
            batch_old_log_probs = []
            batch_probs = []
            batch_returns = []
            batch_advantages = []
            
            # Gather data from all trajectories
            for trajectory in self.trajectories:
                batch_states.extend(trajectory['states'])
                batch_actions.extend(trajectory['actions'])
                batch_old_log_probs.extend(trajectory['action_log_probs'])
                batch_returns.extend(trajectory['returns'])
                batch_advantages.extend(trajectory['advantages'])
                batch_probs.extend(trajectory['probs'])

            # Convert to tensors where appropriate
            batch_actions = torch.stack(batch_actions)
            batch_old_log_probs = torch.stack(batch_old_log_probs)
            batch_returns = torch.tensor(batch_returns, device=device)
            batch_advantages = torch.tensor(batch_advantages, device=device)
            
            # Process in mini-batches to avoid memory issues
            mini_batch_size = self.ppo_minibatch_size
            indices = torch.randperm(len(batch_states))
            
            #for start_idx in range(0, len(batch_states), mini_batch_size):
            # Get mini-batch indices
            mb_indices = indices[0:mini_batch_size]
            
            # Initialize loss components for this mini-batch
            mb_policy_loss = 0
            mb_value_loss = 0
            mb_entropy_loss = 0
            
            # KL Divergence tracking for the batch
            kl_divs = []
            
            # Zero gradients once for the mini-batch
            self.optimizer.zero_grad()
            
            # Process each state in the mini-batch
            for i in mb_indices:
                state_info = batch_states[i]
                
                # Forward pass through policy network
                action_probs, state_value, _, entropy = self.policy_net(
                    state_info['x'].float().detach().requires_grad_(True),
                    state_info['edge_index'].long(),
                    state_info['edge_attr'].float().detach().requires_grad_(True),
                    state_info['question_embeddings'].detach().requires_grad_(True) if state_info['question_embeddings'] is not None else None,
                    subgraph_mask=state_info['subgraph_mask'],
                    action_mask=state_info['action_mask'],
                    action_bias=state_info['action_bias'],
                    question_mask=state_info['question_mask'],
                    use_checkpoint=True
                )
                
                # Get log probability of the action that was taken
                log_prob = torch.log(action_probs + 1e-10).gather(0, batch_actions[i].unsqueeze(0)).squeeze(0)
                old_log_prob = batch_old_log_probs[i]
                
                # Compute KL divergence for just the selected action
                old_prob = batch_probs[i] + 1e-10
                new_prob = action_probs + 1e-10
                
                # KL(old || new) = old_prob * log(old_prob/new_prob)
                with torch.no_grad():
                    kl_state = (old_prob * (torch.log(old_prob) - torch.log(new_prob))).sum()
                kl_divs.append(kl_state)
                
                # Calculate importance sampling ratio
                ratio = torch.exp(log_prob - old_log_prob)
                
                # PPO clipped objective
                clip_param = 0.01
                surr1 = ratio * batch_advantages[i]
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages[i]
                policy_loss_i = -torch.min(surr1, surr2)
                
                # Value function loss
                value_loss_i = F.mse_loss(state_value, batch_returns[i])
                
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
            #loss = mb_policy_loss + 0.5 * mb_value_loss
            # Backward pass and optimization step
            loss.backward()

            grad_norm_custom_gcn = 0.0
            for p in self.policy_net.custom_GCN.parameters():
                if p.grad is not None:
                    grad_norm_custom_gcn += p.grad.data.norm(2).item() ** 2
            grad_norm_custom_gcn = grad_norm_custom_gcn ** 0.5

            print(f"PPO Epoch {epoch}, Custom GCN grad norm: {grad_norm_custom_gcn:.6f}")

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.optimizer.step()
            
            # Log average KL divergence
            if kl_divs:
                avg_kl = sum(kl_divs) / len(kl_divs) 
                print(f"PPO Epoch {epoch}, KL Divergence: {avg_kl:.6f}")
            
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
            'entropy_loss': entropy_loss,
            'valid_steps': 0,
            'valid_trajectories': 0
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
                question_embeddings = state.q_emb,
                action_mask=valid_mask,
                subgraph_mask=state.get_subgraph_mask(),
                question_mask=state.question_mask,
                use_checkpoint=True
            )
            
            # Take most probable valid action
            action = probs.argmax()
            
            # Execute action
            reward, done = state.step(action)
            
            if done:
                break
                
        return state.get_subgraph()
    
    def _batch_update_reinforce(self):
        """Perform REINFORCE update on all trajectories in the buffer"""
        device = next(self.policy_net.parameters()).device
        
        # Log parameters before update
        '''print("\n=== Parameters before update ===")
        for name, param in self.policy_net.named_parameters():
            if param.requires_grad:
                print(f"  {name}: shape={param.shape}")'''
        
        # Original code continues unchanged
        all_returns = []
        valid_trajectories = []
        
        # First pass: calculate returns and filter valid trajectories
        for traj_idx, trajectory in enumerate(self.trajectories):
            # Skip empty trajectories
            if not trajectory['rewards']:
                continue
            
            # Compute returns for this trajectory
            returns = []
            G = 0
            for r in reversed(trajectory['rewards']):
                G = r + 0.99 * G  # gamma = 0.99
                returns.insert(0, G)
            
            # Store returns and collect for normalization
            trajectory['returns'] = returns
            all_returns.extend(returns)
            valid_trajectories.append(trajectory)
        
        # If no valid trajectories found, return early
        if not valid_trajectories:
            print("No valid trajectories found!")
            return {
                'loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'valid_steps': 0,
                'valid_trajectories': 0
            }
        
        # Normalize returns across all trajectories for stability
        if len(all_returns) > 1:
            all_returns_tensor = torch.tensor(all_returns, device=device)
            mean_returns = all_returns_tensor.mean()
            std_returns = all_returns_tensor.std() + 1e-8
            
            for trajectory in valid_trajectories:
                trajectory['returns'] = [(ret - mean_returns) / std_returns for ret in trajectory['returns']]
        
        # ---------- 2. Prepare for policy gradient calculation ----------
        self.optimizer.zero_grad()
        policy_loss = 0.0
        total_valid_steps = 0
        
        # ---------- 3. Accumulate policy gradient loss ----------
        for trajectory_idx, trajectory in enumerate(valid_trajectories):
            # Get proper lengths for this trajectory
            max_steps = min(len(trajectory['states']), len(trajectory['actions']), len(trajectory['returns']))
            
            print(f"\n--- Processing trajectory {trajectory_idx} with {max_steps} steps ---")
            
            for t in range(max_steps):
                state_info = trajectory['states'][t]
                action = trajectory['actions'][t]
                G_t = trajectory['returns'][t]
                
                # CRITICAL FIX: Add these lines to enable gradient tracking on inputs
                # Create copies with requires_grad=True for tensors that need gradients
                x_tensor = state_info['x'].float().detach().requires_grad_(True)
                edge_attr_tensor = state_info['edge_attr'].float().detach().requires_grad_(True)
                q_emb_tensor = state_info['question_embeddings'].detach().requires_grad_(True) if state_info['question_embeddings'] is not None else None
                
                # Log input tensor information
                '''print(f"Step {t} - Input tensors:")
                print(f"  x_tensor requires_grad: {x_tensor.requires_grad}")
                print(f"  edge_attr_tensor requires_grad: {edge_attr_tensor.requires_grad}")
                if q_emb_tensor is not None:
                    print(f"  q_emb_tensor requires_grad: {q_emb_tensor.requires_grad}")'''
                
                # Use these new tensors in the forward pass
                action_probs, _, _, _ = self.policy_net(
                    x_tensor,  # Use the tensor with gradients enabled
                    state_info['edge_index'].long(),  # Indexes don't need gradients
                    edge_attr_tensor,  # Use the tensor with gradients enabled
                    q_emb_tensor,  # Use the tensor with gradients enabled
                    subgraph_mask=state_info['subgraph_mask'],
                    action_mask=state_info['action_mask'],
                    action_bias=state_info['action_bias'],
                    question_mask=state_info['question_mask']
                )
                
                # Log output tensor information
                '''print(f"  action_probs requires_grad: {action_probs.requires_grad}")
                print(f"  action_probs has grad_fn: {action_probs.grad_fn is not None}")'''
                
                # CORE FIX: Use a different approach to extract the log probability
                # This maintains gradient flow properly
                action_idx = action.item()  # Convert to Python scalar
                selected_prob = action_probs[action_idx]
                log_prob = torch.log(torch.clamp(selected_prob, min=1e-10))
                
                # Log probability information
                '''print(f"  log_prob requires_grad: {log_prob.requires_grad}")
                print(f"  log_prob has grad_fn: {log_prob.grad_fn is not None}")'''
                
                # Convert G_t to tensor if needed
                if not isinstance(G_t, torch.Tensor):
                    G_t = torch.tensor(G_t, dtype=torch.float, device=device)
                    
                # Compute loss with proper gradient flow
                step_loss = -log_prob * G_t
                
                # Log step loss information
                '''print(f"  step_loss: {step_loss.item():.6f}")
                print(f"  step_loss requires_grad: {step_loss.requires_grad}")
                print(f"  step_loss has grad_fn: {step_loss.grad_fn is not None}")'''
                
                # Accumulate loss
                policy_loss += step_loss
                total_valid_steps += 1
        
        # ---------- 4. Perform single update ----------
        if total_valid_steps > 0:
            # Average loss over all steps
            policy_loss /= total_valid_steps
            
            # Log final loss before backward
            '''print(f"\n=== Final loss before backward ===")
            print(f"  policy_loss: {policy_loss.item():.6f}")
            print(f"  policy_loss requires_grad: {policy_loss.requires_grad}")
            print(f"  policy_loss has grad_fn: {policy_loss.grad_fn is not None}")'''
            
            # Backward pass (single call for all accumulated gradients)
            policy_loss.backward()
            
            # Log gradients after backward
            '''print("\n=== Gradients after backward ===")
            for name, param in self.policy_net.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
                    else:
                        print(f"  {name}: grad=None")'''
            
            # Gradient clipping for stability
            #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            
            # Log gradients after clipping
            '''print("\n=== Gradients after clipping ===")
            for name, param in self.policy_net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")'''
            
            # Optimization step
            self.optimizer.step()
            
            loss_value = policy_loss.item()
        else:
            loss_value = 0.0
        
        # ---------- 5. Return statistics ----------
        return {
            'loss': loss_value,
            'policy_loss': loss_value,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'valid_steps': total_valid_steps,
            'valid_trajectories': len(valid_trajectories)
        }

def build_pretrain_trajectory(sample, device, directed, triple_graph):
    # Initialize state
    state = RetrievalState(sample, device, directed=directed, triple_graph=triple_graph)

    # Initialize trajectory
    state_list = []
    
    # Get shortest path nodes
    shortest_path_nodes = sample['shortest_path_nodes']
    
    # Create a mask for the true path
    # Make sure it's a boolean tensor, not float
    true_y = torch.zeros(state.graph.edge_index.size(1), dtype=torch.bool, device=device)
    
    # Add edges in the shortest path to the true_y mask
    for i in range(len(shortest_path_nodes)):
        for j in range(len(shortest_path_nodes)):
            
            start_node = shortest_path_nodes[i]
            end_node = shortest_path_nodes[j]
            
            
            # Find edges that match this path segment
            matches = ((state.graph.edge_index[0] == start_node) & 
                    (state.graph.edge_index[1] == end_node))
            
            '''if matches.any():
                #print(f"i: {shortest_path_nodes[i]}, j: {shortest_path_nodes[j]}")'''
            
            # Use logical OR to update the mask
            true_y = true_y | matches
    
    # Get state representation
    valid_mask = torch.ones_like(true_y)
        
    # Store current state
    state_list = {
        'x': state.graph.x.clone().detach(),
        'edge_index': state.graph.edge_index.detach(),
        'edge_attr': state.graph.edge_attr.detach(),
        'question_embeddings': state.q_emb.clone().detach(),
        'question_mask': state.question_mask.clone(),
        'y': valid_mask.clone() & true_y.clone()
    }
    
    return state_list



        