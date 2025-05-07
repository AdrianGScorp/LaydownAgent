"""
Deep Q-Learning agent for warehouse optimization.

This module implements a DQN agent with experience replay and transformer-based
Q-networks for optimal item group selection.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from models import AttentionQNetwork


class DQNAgent:
    """Deep Q-Learning agent with experience replay and dynamic exploration.
    
    Key Features:
        - Epsilon-greedy exploration (decays exponentially over time)
        - Double DQN architecture with agent and target network
        - Masked Q-values to prevent invalid actions (selecting already used items)
    
    Attributes:
        model (AttentionQNetwork): Main Q-network for action selection and updates.
        target (AttentionQNetwork): Target Q-network for stable learning.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        memory (deque): Experience replay buffer.
        epsilon (float): Current exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay factor for exploration rate.
        gamma (float): Discount factor for future rewards.
        device (torch.device): Device for computation (CPU/GPU).
        action_type (str): Type of the last action taken ('random' or 'policy').
    """
    
    def __init__(self, num_q_dims: int, epsilon_min: float, epsilon_decay: float, 
                 learning_rate: float, gamma: float, device: torch.device,
                 hidden_dim: int, n_heads: int, n_layers: int):
        """Initialize the DQN agent.
        
        Args:
            num_q_dims: Number of quality dimensions per item.
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Decay factor for exploration rate.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor for future rewards.
            device: Device for computation.
            hidden_dim: Size of hidden layers in networks.
            n_heads: Number of attention heads in transformer.
            n_layers: Number of transformer layers.
        """
        self.device = device
        feature_dim = num_q_dims + 4  # qualities + (Availability + 3 temporal features)
        
        # Initialize networks
        self.model = AttentionQNetwork(feature_dim, hidden_dim, n_heads, n_layers).to(device)
        self.target = AttentionQNetwork(feature_dim, hidden_dim, n_heads, n_layers).to(device)
        self.target.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0  # Initial exploration rate 
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.action_type = None

    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.
        
        Masks invalid actions (selecting unavailable items) by setting
        their Q-values to -inf.
        
        Args:
            state: Current environment state.
            
        Returns:
            int: Selected action (item index) or None if no valid actions.
        """
        available = np.where(state[:, 0] == 1)[0]
        if not available.size:
            return None

        if np.random.rand() < self.epsilon:
            self.action_type = 'random'
            return np.random.choice(available)
        else:
            self.action_type = 'policy'            
            return self._policy_action(state, available)

    def _policy_action(self, state: np.ndarray, available: np.ndarray) -> int:
        """Select action using current Q-network.
        
        Args:
            state: Current environment state.
            available: Indices of available items.
            
        Returns:
            int: Selected action (item index).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy().flatten()
            
        # Mask unavailable actions
        q_values = np.where(state[:, 0] == 1, q_values, -np.inf)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size: int) -> float:
        """Train network on experience replay memory.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            float: Training loss value or None if memory is insufficient.
        """
        if len(self.memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self._sample_batch(batch_size)
        current_q = self.model(states).gather(1, actions).squeeze()
        target_q = self._calculate_targets(rewards, next_states, dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._decay_epsilon()
        return loss.item()

    def _sample_batch(self, batch_size: int) -> tuple:
        """Sample and format experience batch.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            tuple: Batch of experiences as tensors.
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        states_np = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=bool)
        
        return (
            # Convert to tensors and move to device
            torch.from_numpy(states_np).to(self.device),
            torch.from_numpy(actions_np).unsqueeze(-1).to(self.device),
            torch.from_numpy(rewards_np).to(self.device),
            torch.from_numpy(next_states_np).to(self.device),
            torch.from_numpy(dones_np).to(self.device)
        )

    def _calculate_targets(self, rewards: torch.Tensor, 
                          next_states: torch.Tensor, 
                          dones: torch.Tensor) -> torch.Tensor:
        """Compute target Q-values using target network.
        
        Uses the Double DQN approach to reduce overestimation bias.
        
        Args:
            rewards: Batch of rewards.
            next_states: Batch of next states.
            dones: Batch of done flags.
            
        Returns:
            torch.Tensor: Target Q-values.
        """
        with torch.no_grad():
            # Main network selects actions
            main_net_q = self.model(next_states)  # Q-values from main network to select actions
            mask = next_states[:, :, 0] == 1
            main_net_q[~mask] = -float('inf')
            best_actions = main_net_q.argmax(dim=1, keepdim=True)  # Action selection

            # Target network evaluates the selected actions
            target_net_q = self.target(next_states)  # Q-values from target network for future rewards
            max_next_q = target_net_q.gather(1, best_actions).squeeze()  # Evaluation

            # Handle terminal states
            max_next_q[torch.isinf(max_next_q)] = 0.0
            
        return rewards + self.gamma * max_next_q * (~dones)

    def _decay_epsilon(self):
        """Exponentially decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """Update target network with current model parameters."""
        self.target.load_state_dict(self.model.state_dict())
        
    def save(self, filepath):
        """Save agent model and parameters.
        
        Args:
            filepath: Path to save the model.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, filepath)
        
    def load(self, filepath, device=None):
        """Load agent model and parameters.
        
        Args:
            filepath: Path to the saved model.
            device: Device to load the model to.
        """
        if device:
            self.device = device
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
