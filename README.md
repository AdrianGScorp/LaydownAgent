# Warehouse Group Optimization with Deep Reinforcement Learning

## Project Overview
This project trains a Deep Q-Network (DQN) agent to form homogeneous groups of items in a warehouse environment. The agent selects items with similar quality attributes over consecutive steps, aiming to minimize variance within groups and maintain consistency between groups over time.

## Key Features
### Core Architecture
- **Transformer-based Q-Network**: Self-attention mechanism captures complex item relationships  
- **Double DQN**: training two Q-Networks for more stable learning
  - Main Q-network for action selection.
  - Target Q-network for predicting Q-values (predicted future reward) for selected action and resulting state. (updated less frequently)
- **Temporal Features**: Each item's state includes dynamically computed features (deviation from previous group mean, normalized episode progress, and cumulative group variance), guiding the agent's decision-making over time.
- **Synthetic Data**: Synthetic Warehous generation based on random clusters and gausian noise for quality parameters.  
### Learning Strategy
- **Curriculum Learning**: Gradually increases maximum allowed steps per episode to create harder scenarios.
- **Dynamic Exploration**: Epsilon-greedy strategy with exponential decay
- **Dual Reward System**:
  - *Intra-group penalty*: Variance within current group
  - *Inter-group penalty*: Difference from previous group means
### Optimization
- **Hyperparameter Tuning**: Optuna integration  
- **WandB Logging**: Tracks rewards, epsilon decay, and training speed
- **GPU Acceleration**: CUDA support for transformer computations

## Synthetic Data Generation
Because there was no real data available, the warehouse environment uses artificially generated item quality data to simulate realistic grouping challenges. This is done via the `generate_data()` function which generates clustered item qualities with the following main steps:
- **Cluster Centers**: Randomly generate `n_clusters` (5, fix default in code) centers in the quality space.
- **Item Assignment**: Items are evenly assigned to these clusters.
- **Noise Injection**: Add small Gaussian noise to each item to simulate real-world variation.
- **Normalization**: Clip values to [0, 1] and shuffle to randomize order.

To better visualize the clustering process used, the following figure shows a simple example with three clusters along a single quality dimension:

![image](https://github.com/user-attachments/assets/eabaafbc-fb09-4e7f-9791-b63da66337fa)

#### Data Structure
- `qualities`: `np.ndarray` of shape `(num_items, num_q_dims)`  
  Each row represents an item's quality vector across dimensions.
- `availability`: `np.ndarray` of shape `(num_items,)`  
  Binary vector where 1 = available for selection, 0 = already selected.

### Episode Flow
Each episode consists of the following loop:
1. **Initialize the environment**  
   - Generate new synthetic item data.  
   - Mark all items as available.  
   - Reset all internal states and counters.
2. **Repeat for each step (up to `max_steps` or until not enough items remain):**  
   a. Observe the current environment state.  
   b. Agent selects one item to add to the current group.  
   c. Environment marks the item as unavailable.  
   d. If `k` items have been selected:  
      - Form a group.  
      - Calculate the reward based on intra-group variance and consistency with past groups.  
      - Record group statistics.  
      - Reset the current group.  
   e. If an invalid action is taken (item already selected), the episode ends immediately.  
3. **Episode ends**  
   - When `max_steps` is reached or  
   - When fewer than `k` available items remain.  
   - Then, availability is reset and all episode-specific states are cleared:
     - `episode_means`, `episode_vars`, and `current_group` are emptied.  
     - `completed_steps` is reset to 1.  
     - `group_completed` is set to False.  
     - **Temporal features** (deviation, progress, cumulative variance) are implicitly reset, as they are derived from the cleared variables.

---

## Todo's
- get an idea of the performance:
  - Compare the agent to a random selector
  - Compare to existing solutions
  - Use real Warehouse data
- Investigate further with different architectures:
  - Instead of DQN try PPO (proximal policy optimisation) algorithem
- improfe synthetic data creation
  - distribute items randomly across clusters (equally distribution at the moment). Closer to reality and harder.
