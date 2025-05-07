"""
Main script for warehouse optimization agent training.

This script orchestrates the training process for the DQN agent
in the warehouse environment.
"""
import torch
from hyperparameter_tuning import optimize_hyperparameters
from training import run_training


def main():
    """Main function for running the warehouse optimization training.
    
    Controls the overall flow of the training process, including:
    1. Hyperparameter optimization (if enabled)
    2. Final training with best parameters
    """    
    run_name = "items100-q5-k5" 
    # Whether to run hyperparameter optimization
    optimize = True
    
    # Base training arguments
    training_args = {
        'num_episodes': (1000 if optimize else 2000),  # Shorter for optimization runs
        'batch_size': 32,
        'init_max_steps': 20,
        'num_items': 100,
        'num_q_dims': 5,
        'k': 5,  # Number of items per group
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 10,
        'max_steps_update_freq': 2000,
        'hidden_dim': 64,
        'n_heads': 4,
        'n_layers': 2,
        'reward_function': 'mdv',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # Run hyperparameter optimization if enabled
    if optimize:
        print("Starting hyperparameter optimization...")
        best_trial_number, best_params = optimize_hyperparameters(training_args, run_name=run_name, n_trials=50)
        
        # Update training arguments with best parameters
        for param, value in best_params.items():
            training_args[param] = value
        
        print(f"Optimization complete. Running final training with best parameters from trial-{best_trial_number}.")
    
    # Run final training with best parameters
    print("Starting training with the following parameters:")
    for param, value in training_args.items():
        print(f"  {param}: {value}")
    
    # Run the training
    avg_reward = run_training(training_args, run_name=run_name, save_agent=True)
    
    print(f"Training completed. Final average reward: {avg_reward:.4f}")


if __name__ == "__main__":
    main()