# Warehouse Group Optimization with Deep Reinforcement Learning

## Project Overview
This project trains a Deep Q-Network (DQN) agent to form homogeneous groups of items in a warehouse environment. The agent selects items with similar quality attributes over consecutive steps, aiming to minimize variance within groups and maintain consistency between groups over time.

## Key Features
- **Transformer-based Q-Network**: Captures complex item relationships through self-attention mechanisms.
- **Curriculum Learning**: Gradually increases maximum allowed steps per episode to create harder scenarios.
- **Dual Reward System**:
  - *Intra-group penalty*: Variance within current group
  - *Inter-group penalty*: Difference from previous group means
- **Dynamic Exploration**: Epsilon-greedy strategy with exponential decay
- **Optuna Integration**: Automated hyperparameter optimization

---

## Todo's
- get an idea of the performance:
  - Compare the agent to a random selector
  - Compare to existing solutions
  - Use real Warehouse data
- Investigate further with different architectures:
  - Instead of DQN try PPO (proximal policy optimisation) algorithem
