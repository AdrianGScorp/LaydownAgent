"""
Training utilities for the warehouse optimization agent.

This module contains functions for training the DQN agent and
running experiments with different hyperparameters.
"""
import numpy as np
import torch
import time
from collections import deque
import wandb
from environment import WarehouseEnvironment
from agent import DQNAgent


def run_training(training_args, run_name=None, save_agent=False):
    """Execute the training loop and return the average total reward.
    
    Args:
        training_args: Dictionary of training arguments.
        run_name: Name for the training run (for logging).
        save_agent: Whether to save the trained agent.
        
    Returns:
        float: Average total reward over the last episodes.
    """
    # Setup wandb logging if desired
    if run_name:
        wandb.init(project="WarehouseAgent", name=run_name, config=training_args)

    # Initialize environment and agent
    env = WarehouseEnvironment(
        num_items=training_args['num_items'],
        num_q_dims=training_args['num_q_dims'],
        k=training_args['k'],
        init_max_steps=training_args['init_max_steps'],
        reward_function=training_args['reward_function']
    )
    agent = DQNAgent(
        num_q_dims=training_args['num_q_dims'],
        epsilon_min=training_args['epsilon_min'],
        epsilon_decay=training_args['epsilon_decay'],
        learning_rate=training_args['learning_rate'],
        gamma=training_args['gamma'],
        hidden_dim=training_args['hidden_dim'],
        n_heads=training_args['n_heads'],
        n_layers=training_args['n_layers'],
        device=training_args['device']
    )

    # Log agent parameter count if using wandb
    if run_name:
        wandb.config.update({"agent_num_params": agent.model.count_parameters()})

    total_rewards = []
    window_size = 5
    recent_durations = deque(maxlen=window_size)
    steps_limit = training_args['num_items'] // training_args['k']

    # Main training loop
    for episode in range(training_args['num_episodes']):
        start_time = time.time()
        state = env.initialize_env()
        total_reward = 0
        done = False
        action_counts = {'random': 0, 'policy': 0}

        while not done:
            action = agent.act(state)
            if action is None:
                break
                
            next_state, reward, done = env.step(action)
            if env.group_completed:
                env.reset_group()
                
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            action_counts[agent.action_type] += 1
        
        # Calculate running average of total rewards
        total_rewards.append(total_reward)
        avg_total_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
        
        # Train the agent
        loss = agent.train(training_args['batch_size'])
        
        # Track training speed metrics
        recent_durations.append(time.time() - start_time)
        curr_train_speed = len(recent_durations) / (sum(recent_durations) + 1e-9)

        # Log metrics
        log_data = {
            "total_reward": total_reward,
            "avg_total_reward": avg_total_reward,
            "epsilon": agent.epsilon,
            "completed_steps": env.completed_steps,
            "max_steps": env.max_steps,
            "num_random_actions": action_counts['random'],
            "num_policy_actions": action_counts['policy'],
            "episode_duration": recent_durations[-1],
            "train_speed": curr_train_speed,
        }
        if loss is not None:
            log_data["loss"] = loss

        if run_name:
            wandb.log(log_data, step=episode)

        # Update target network periodically
        if (episode + 1) % training_args['target_update_freq'] == 0:
            agent.update_target_network()
            if run_name:
                wandb.log({"target_updated": 1}, step=episode)
        else:
            if run_name:
                wandb.log({"target_updated": 0}, step=episode)

        # Curriculum learning: Increase max_steps to increase difficulty
        if (episode + 1) % training_args['max_steps_update_freq'] == 0 and env.max_steps < steps_limit:
            env.max_steps += 1
            agent.epsilon = 1.0  # reset exploration
            if run_name:
                wandb.log({"max_steps_increased": 1}, step=episode)
        else:
            if run_name:
                wandb.log({"max_steps_increased": 0}, step=episode)

        # Print progress
        print(f"Ep {episode:5d} | Reward: {total_reward:7.3f} | "
              f"Epsilon: {agent.epsilon:.2f} | Completed steps: {env.completed_steps:3d} | "
              f"Speed: {curr_train_speed:.1f} eps/s")

    if run_name:
        wandb.finish()
    print("Training completed.")
    
    if save_agent:
        agent.save(f"agents//agent_{run_name}.pth")
        print(f"Agent saved to agent_{run_name}.pth")

    # Calculate average reward of last 100 episodes
    avg_total_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
    return avg_total_reward
