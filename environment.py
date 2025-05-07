"""
Environment module for warehouse optimization simulation.

This module implements a warehouse environment for optimizing group selection
of items with similar quality attributes.
"""
import numpy as np


class WarehouseEnvironment:
    """Simulates a warehouse environment for optimizing group selection of items.
    
    The environment supports curriculum learning by allowing the maximum number
    of steps to increase over time, creating progressively harder scenarios.
    
    Attributes:
        num_items (int): Total number of items in the warehouse.
        num_q_dims (int): Number of quality dimensions/variables per item.
        k (int): Required group size for selection.
        max_steps (int): Maximum steps allowed per episode.
            Increased during training to create harder environments.
        qualities (np.ndarray): Array of shape (num_items, num_q_dims) storing item quality scores.
        availability (np.ndarray): Binary array indicating item availability (1=available, 0=selected).
        episode_means (list): List of mean quality vectors for completed groups in current episode.
        current_group (list): List of indices of currently selected items.
        completed_steps (int): Number of completed steps in current episode.
        group_completed (bool): Flag indicating if the current group is complete.
    """
    
    def __init__(self, num_items: int, num_q_dims: int, k: int, init_max_steps: int = 1, reward_function: str = "mdv"):
        """Initialize the warehouse environment.
        
        Args:
            num_items: Total number of items in the warehouse.
            num_q_dims: Number of quality dimensions per item.
            k: Number of items required per group.
            init_max_steps: Initial value for maximum steps per episode.
        """
        self.num_items = num_items
        self.num_q_dims = num_q_dims
        self.k = k  # Items required per group
        self.max_steps = init_max_steps
        self.episode_means = []
        self.episode_vars = []
        self.qualities = np.empty((num_items, num_q_dims), dtype=np.float32)
        self.availability = np.empty(num_items, dtype=np.float32)
        self.current_group = []
        self.completed_steps = 0
        self.group_completed = False 
        self.reward_function = reward_function
        self.initialize_env() # Generate initial data and reset state

    def initialize_env(self):
        """Reset environment for new episode with new data."""
        self.generate_data()
        self.reset_state()
        return self._get_state()
    
    def reset_state(self):
        """Reset selection of items for new episode with same data."""
        self.episode_means = []
        self.episode_vars = []
        self.current_group = []
        self.completed_steps = 1
        self.availability[:] = 1.0
        self.group_completed = False
        return self._get_state()

    def generate_data(self, seed: int | None = None):
        """Generate clustered item qualities with Gaussian noise.
        
        Creates quality values in clusters to encourage the agent to learn
        to select items from similar quality clusters.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Generate clustered data
        n_clusters = 5
        centers = np.random.rand(n_clusters, self.num_q_dims).astype(np.float32)
        cluster_size = self.num_items // n_clusters
        repeats = [cluster_size] * n_clusters
        repeats[-1] += self.num_items % n_clusters
        
        self.qualities = np.repeat(centers, repeats, axis=0)
        self.qualities += np.random.normal(scale=0.01, size=self.qualities.shape).astype(np.float32)
        np.clip(self.qualities, 0.0, 1.0, out=self.qualities)
        np.random.shuffle(self.qualities)

    def _get_state(self) -> np.ndarray:
        """Compile the environment state.
        
        Returns:
            np.ndarray: State tensor of shape (num_items, num_q_dims + 4) containing:
                - Column [0]: Item availability
                - Columns [1:1+num_q_dims]: Item quality scores
                - Remaining columns: Temporal features (deviation from previous group mean, 
                                     episode progress, cumulative variance).
        """
        base_features = np.hstack([self.availability.reshape(-1, 1), self.qualities])
        temporal_features = self._create_temporal_features()
        return np.hstack([base_features, temporal_features]).astype(np.float32)

    def _create_temporal_features(self) -> np.ndarray:
        """Generate time-dependent features for state representation.
        
        Creates features that provide context about the episode history to
        guide the agent in making consistent selections across groups.

        Features:
            1. Deviation: Mean absolute difference from the last group's quality mean.
            2. Progress: Normalized remaining steps in the episode.
            3. Cumulative Variance: Historical variance of all group means in the episode.
            
        Returns:
            np.ndarray: Array of temporal features for each item.
        """
        features = []
        
        # Deviation from previous group mean (per item)
        if self.episode_means:
            previous_means = self.episode_means[-1]
            deviation = np.mean(np.abs(self.qualities - previous_means), axis=1, keepdims=True)
        else:
            deviation = np.zeros((self.num_items, 1), dtype=np.float32)
        features.append(deviation)

        # Episode progress
        progress = np.full((self.num_items, 1), (self.max_steps - self.completed_steps)/self.max_steps)
        features.append(progress)

        # Cumulative variance (sum of variances across dimensions)
        if self.episode_means:
            means_array = np.array(self.episode_means)
            cum_var_per_qdim = np.var(means_array, axis=0)
            cum_var = np.sum(cum_var_per_qdim)
        else:
            cum_var = 0.0
        features.append(np.full((self.num_items, 1), cum_var))
        
        return np.hstack(features)

    def step(self, action: int) -> tuple:
        """Execute one environment step.
        
        Args:
            action: Item index to select.
            
        Returns:
            tuple: (next_state, reward, done) where:
                - next_state: Updated environment state.
                - reward: Reward for the action taken.
                - done: Boolean indicating if episode is complete.
        """
        if self.availability[action] != 1:
            # Invalid action: Item is already selected, return punishment and end episode
            return self._get_state(), -100, True

        self.current_group.append(action)
        self.availability[action] = 0

        if len(self.current_group) < self.k:
            # Group is not complete yet, continue with no reward
            return self._get_state(), 0, False
        
        # Group is complete, calculate reward and prepare for next group
        next_state, reward, done = self._handle_completed_group()
        self.group_completed = True           
        return next_state, reward, done
    
    def reset_group(self):
        """Reset the current group after group is complete.
        
        Called after group completion for visualization and logging purposes.
        """
        self.current_group = []
        self.group_completed = False

    def _handle_completed_group(self) -> tuple:
        """Calculate rewards and update state after selecting k items.
        
        The reward reflects the homogeneity within and across groups.
        Higher reward indicates better homogeneity.
        
        Reward Formula:
            reward = -1 * (group_variance + mean_diff)
            - group_variance: Variance of qualities in current group (lower = better uniformity)
            - mean_diff: Weighted difference from previous group trends (lower = better consistency)
        
        Returns:
            tuple: (next_state, reward, done)
        """
        group_qualities = self.qualities[self.current_group]
        current_means = np.mean(group_qualities, axis=0) # mean of each quality dim from the current group
        current_vars = np.var(group_qualities, axis=0) # variance of each quality dim from the current group
        self.episode_means.append(current_means)
        self.episode_vars.append(current_vars)
        
        reward = self._calculate_reward()

        # # Calculate sum of variance per dimension
        # variances_per_dim = np.var(group_qualities, axis=0)
        # sum_variances = np.sum(variances_per_dim)

        # # Current group's mean (vector of per-dimension means)
        # current_means = np.mean(group_qualities, axis=0)

        # # Mean difference (sum of absolute differences across dimensions)
        # if self.episode_means:
        #     previous_means = self.episode_means[-1]
        #     sum_mean_diff = np.sum(np.abs(current_means - previous_means))
        # else:
        #     sum_mean_diff = 0.0

        # # Calculate weighted average difference from previous trends
        # if self.episode_means:
        #     # Previous weighted average (old_avg: up to last 5 means)
        #     lookback_window = min(len(self.episode_means), 5)
        #     avg_weights = np.arange(1, lookback_window + 1)
        #     avg_weights = avg_weights / avg_weights.sum()

        #     old_window = self.episode_means[-lookback_window:]
        #     old_avg = np.sum(np.array(old_window) * avg_weights.reshape(-1, 1), axis=0)

        #     # New weighted average (new_avg: up to last 4 means + current)
        #     new_window = self.episode_means[-max(0, lookback_window-1):] + [current_means]
        #     new_avg = np.sum(np.array(new_window) * avg_weights.reshape(-1, 1), axis=0)

        #     # Penalize deviation between old and new trends
        #     sum_mean_diff = np.sum(np.abs(new_avg - old_avg))
        # else:
        #     sum_mean_diff = 0.0
        
        # reward = -1 * (sum_variances + sum_mean_diff)
        
        # Episode ends if we've reached max steps or if there are not enough items left
        done = (self.completed_steps >= self.max_steps or 
                np.sum(self.availability) < self.k)
                
        # Increment step counter if episode continues
        if not done:
            self.completed_steps += 1
        
        return self._get_state(), reward, done
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on the configured reward function.

        Args:
            group_qualities: Qualities of the items in the current group.
            current_means: Mean quality vector of the current group.

        Returns:
            The calculated reward value.
        """
        if self.reward_function == "mdv":
            return self._reward_mdv()
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")

    def _reward_mdv(self) -> float:
        """Default reward function: -1 * (group_variance + mean_diff)."""
        # Calculate sum of variance per dimension
        sum_variances = np.sum(self.episode_vars[-1])

        # Mean difference (sum of absolute differences across dimensions)
        if len(self.episode_means)>=2:
            previous_means = self.episode_means[-2]
            current_means = self.episode_means[-1]
            sum_mean_diff = np.sum(np.abs(current_means - previous_means))
        else:
            sum_mean_diff = 0.0
        
        return -1 * (sum_variances + sum_mean_diff)
