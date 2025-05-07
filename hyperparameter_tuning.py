"""
Hyperparameter optimization for the warehouse optimization agent.

This module contains functions for optimizing hyperparameters using Optuna.
"""
import optuna
from training import run_training


def objective(trial, training_args=None, run_name=None):
    """Optuna objective function to optimize hyperparameters.
    
    Args:
        trial: Optuna trial object.
        training_args: Base training arguments to modify.
        
    Returns:
        float: Average reward to maximize.
    """
    # Create a copy of training_args to avoid modifying the original
    trial_args = training_args.copy() if training_args else {}
    
    # Hyperparameter suggestions
    trial_args["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    trial_args["gamma"] = trial.suggest_float("gamma", 0.8, 0.99)
    trial_args["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    trial_args["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
    trial_args["n_layers"] = trial.suggest_int("n_layers", 1, 3)
    trial_args["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    trial_args["target_update_freq"] = trial.suggest_int("target_update_freq", 5, 50, step=5)
    trial_args["epsilon_decay"] = trial.suggest_float("epsilon_decay", 0.99, 0.999)

    # Run training with trial parameters
    avg_reward = run_training(
        trial_args, 
        run_name=f"{run_name}_optuna-trial-{trial.number}"
    )
    
    return avg_reward


def optimize_hyperparameters(training_args, run_name=None, n_trials=50):
    """Run hyperparameter optimization study.
    
    Args:
        training_args: Base training arguments.
        n_trials: Number of optimization trials.
        
    Returns:
        dict: Best hyperparameters found.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, training_args, run_name), n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Avg Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.number, study.best_params
