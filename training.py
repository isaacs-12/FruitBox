import gymnasium as gym
from gymnasium import spaces
import numpy as np
import argparse
import os
from datetime import datetime
import pickle
import random
from typing import List, Tuple, Optional
import time
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from training_data import TrainingDataCollector, collect_training_data_from_model

# --- 1. Define the Custom Reinforcement Learning Environment ---

class RectangleEnv(gym.Env):
    """Custom Environment for finding rectangles that sum to 10."""
    
    def __init__(self, initial_grid=None, render_mode=None, grids=None):
        super().__init__()
        self.grids = grids  # List of grids for training
        self.current_grid_idx = 0
        
        if initial_grid is not None:
            self.grid_shape = initial_grid.shape
        else:
            # Default to a 10x17 grid if no initial grid provided
            self.grid_shape = (10, 17)
        
        self.rows, self.cols = self.grid_shape
        self.observation_space = spaces.Box(low=0, high=9, shape=(self.rows, self.cols, 1), dtype=np.uint8)
        self.max_possible_actions = ((self.rows * (self.rows + 1) // 2) * (self.cols * (self.cols + 1) // 2))
        self.action_space = spaces.Discrete(self.max_possible_actions)
        self.initial_grid = initial_grid if initial_grid is not None else self._generate_random_grid()
        self.grid = None
        self.rectangle_id_map = {}
        self._populate_rectangle_id_map()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.last_action = None
        self.last_reward = None
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_digits_cleared = 0
        self.initial_digits = 0  # Track initial number of non-zero digits
    
    def _generate_random_grid(self):
        """Generates a random initial grid with digits 1-9."""
        return np.random.randint(1, 10, size=self.grid_shape, dtype=np.uint8)

    def _populate_rectangle_id_map(self):
        """
        Populates a map from integer IDs to (r1, c1, r2, c2) tuples.
        This allows a fixed discrete action space to map to specific rectangles.
        """
        current_id = 0
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                for r2 in range(r1, self.rows):
                    for c2 in range(c1, self.cols):
                        self.rectangle_id_map[current_id] = (r1, c1, r2, c2)
                        current_id += 1
        # This asserts that max_possible_actions is correctly calculated
        assert current_id == self.max_possible_actions, f"Mismatch in action count: {current_id} vs {self.max_possible_actions}"

    def _get_rectangle_from_action(self, action):
        """Convert action index to rectangle coordinates"""
        if isinstance(action, np.ndarray):
            action = action.item()  # Convert numpy array to scalar
        return self.rectangle_id_map[action]
    
    def _get_valid_actions(self):
        """Get mask of valid actions."""
        valid_actions = np.zeros(self.max_possible_actions, dtype=bool)
        for action_id, (r1, c1, r2, c2) in self.rectangle_id_map.items():
            # Check if rectangle is valid (sums to 10 and has at least one non-zero)
            rect_sum = 0
            has_non_zero = False
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if self.grid[r, c] != 0:
                        rect_sum += self.grid[r, c]
                        has_non_zero = True
            if has_non_zero and rect_sum == 10:
                valid_actions[action_id] = True
        return valid_actions
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.grids is not None:
            # Cycle through training grids
            self.initial_grid = self.grids[self.current_grid_idx]
            self.current_grid_idx = (self.current_grid_idx + 1) % len(self.grids)
        else:
            self.initial_grid = self._generate_random_grid()
        
        self.grid = np.copy(self.initial_grid)
        self.initial_digits = np.count_nonzero(self.grid)  # Track initial non-zero digits
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_digits_cleared = 0
        self.last_action = None
        self.last_reward = None
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.grid.reshape(self.rows, self.cols, 1), {"action_mask": self._get_valid_actions()}
    
    def step(self, action):
        r1, c1, r2, c2 = self._get_rectangle_from_action(action)
        
        # Count non-zero digits before clearing
        digits_before = np.count_nonzero(self.grid)
        
        # Zero out the rectangle
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.grid[r, c] = 0
        
        # Count non-zero digits after clearing
        digits_after = np.count_nonzero(self.grid)
        digits_cleared = digits_before - digits_after
        
        # Calculate reward based on digits cleared
        # Base reward is the number of digits cleared
        reward = digits_cleared
        
        # Add a bonus for clearing a high percentage of remaining digits
        if digits_before > 0:
            clear_percentage = digits_cleared / digits_before
            # Bonus scales with how much of the remaining digits were cleared
            # e.g., clearing 50% of remaining digits gives a 0.5x bonus
            reward *= (1 + clear_percentage)
        
        # Add a completion bonus if we clear all digits
        if digits_after == 0 and self.initial_digits > 0:
            completion_bonus = self.initial_digits * 2  # Large bonus for clearing everything
            reward += completion_bonus
        
        # Update episode tracking
        self.episode_reward += reward
        self.episode_length += 1
        self.episode_digits_cleared += digits_cleared
        self.last_action = (r1, c1, r2, c2)
        self.last_reward = reward
        
        # Check if done (no more valid actions)
        done = not np.any(self._get_valid_actions())
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.grid.reshape(self.rows, self.cols, 1), reward, done, False, {"action_mask": self._get_valid_actions()}

    def render(self):
        """
        Renders the current state of the grid.
        For simplicity, prints to console.
        """
        if self.render_mode == "human":
            print("\nCurrent Grid State:")
            print(self.grid)
            print(f"Rectangles found so far: {self.episode_length}")
            if self.last_action:
                print(f"Last action: {self.last_action}")
            print("-" * 20)

    def close(self):
        pass # No resources to close


# --- 2. Callback for Logging and Action Masking with Stable Baselines3 ---

# Stable Baselines3's MaskablePPO automatically uses the "action_mask" in the info dict
# if you provide it in the environment's step and reset methods.
# No special callback is strictly needed for the masking itself if set up correctly.

# You might want a custom callback for custom logging or stopping criteria:
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, val_grids=None, verbose=0, eval_freq=10000, eval_episodes=3, eval_verbose=False):
        super().__init__(verbose)
        self.val_grids = val_grids
        self.best_mean_reward = -np.inf
        self.best_model_path = None
        self.eval_freq = eval_freq  # Evaluate every N steps (default: 10k instead of 5k)
        self.eval_episodes = eval_episodes  # Number of episodes per evaluation (default: 3 instead of 5)
        self.eval_verbose = eval_verbose  # Whether to print detailed evaluation progress
        self.last_eval_step = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.print_freq = 10  # Print progress every 10 seconds
        self.last_step_time = time.time()
        self.step_timeout = 300  # Consider it hung if no step for 5 minutes (increased from 30 seconds)
        
        # Training data collection for plotting
        self.training_data = {
            'timesteps': [],
            'mean_rewards': [],
            'eval_timesteps': [],
            'eval_mean_rewards': [],
            'episode_lengths': [],
            'digits_cleared': []
        }
        self.last_eval_reward = None
        self.model_timestamp = None  # Will be set when model is saved

    def _on_step(self) -> bool:
        try:
            current_time = time.time()
            
            # Check for hanging
            if current_time - self.last_step_time > self.step_timeout:
                print(f"\nWARNING: No step completed for {self.step_timeout} seconds!")
                print("Current state:")
                print(f"  Steps: {self.num_timesteps:,}/{self.locals.get('total_timesteps', 0):,}")
                print(f"  Last eval step: {self.last_eval_step:,}")
                print(f"  Time since last step: {current_time - self.last_step_time:.1f} seconds")
                if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                    print(f"  Last episode reward: {self.model.ep_info_buffer[-1]['r']:.2f}")
                return False  # Stop training if we detect a hang
            
            self.last_step_time = current_time
            
            # Print progress every 10 seconds
            if current_time - self.last_print_time >= self.print_freq:
                elapsed_time = current_time - self.start_time
                steps_per_second = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
                remaining_steps = self.locals.get('total_timesteps', 0) - self.num_timesteps
                remaining_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                
                print(f"\nProgress: {self.num_timesteps:,}/{self.locals.get('total_timesteps', 0):,} steps "
                      f"({(self.num_timesteps/self.locals.get('total_timesteps', 1)*100):.1f}%)")
                print(f"Speed: {steps_per_second:.1f} steps/second")
                print(f"Elapsed: {elapsed_time/60:.1f} minutes")
                print(f"Remaining: {remaining_time/60:.1f} minutes")
                
                # Print current episode stats if available
                if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                    mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                    mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                    print(f"Current mean reward: {mean_reward:.2f}")
                    print(f"Current mean episode length: {mean_length:.1f}")
                    
                    # Collect training data for plotting
                    self.training_data['timesteps'].append(self.num_timesteps)
                    self.training_data['mean_rewards'].append(mean_reward)
                    self.training_data['episode_lengths'].append(mean_length)
                
                self.last_print_time = current_time
            
            # Evaluate periodically
            if self.num_timesteps - self.last_eval_step >= self.eval_freq:
                print(f"\nStarting validation at step {self.num_timesteps:,}...")
                eval_start_time = time.time()
                self.last_eval_step = self.num_timesteps
                
                try:
                    if self.val_grids is not None:
                        # Evaluate on validation grids
                        mean_reward = self._evaluate_on_grids(self.val_grids)
                        eval_time = time.time() - eval_start_time
                        print(f"Validation completed in {eval_time:.1f} seconds")
                        print(f"Mean reward: {mean_reward:.2f}")
                        
                        # Save best model
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            if self.best_model_path is not None:
                                os.remove(self.best_model_path)
                            self.best_model_path = f"models/best_model_{int(time.time())}.zip"
                            self.model.save(self.best_model_path)
                            print(f"New best model saved to {self.best_model_path}")
                        
                        # Collect evaluation data for plotting
                        self.training_data['eval_timesteps'].append(self.num_timesteps)
                        self.training_data['eval_mean_rewards'].append(mean_reward)
                        self.last_eval_reward = mean_reward
                    else:
                        # Original evaluation on random grids
                        mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                        eval_time = time.time() - eval_start_time
                        print(f"Evaluation completed in {eval_time:.1f} seconds")
                        print(f"Mean reward: {mean_reward:.2f}")
                        
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            if self.best_model_path is not None:
                                os.remove(self.best_model_path)
                            self.best_model_path = f"models/best_model_{int(time.time())}.zip"
                            self.model.save(self.best_model_path)
                            print(f"New best model saved to {self.best_model_path}")
                        
                        # Collect evaluation data for plotting
                        self.training_data['eval_timesteps'].append(self.num_timesteps)
                        self.training_data['eval_mean_rewards'].append(mean_reward)
                        self.last_eval_reward = mean_reward
                except Exception as e:
                    print(f"\nERROR during validation: {str(e)}")
                    print("Continuing training...")
            
            return True
            
        except Exception as e:
            print(f"\nERROR in training callback: {str(e)}")
            return False  # Stop training on error
        
    def _evaluate_on_grids(self, grids, n_episodes=None):
        """Evaluate model on a set of grids"""
        if n_episodes is None:
            n_episodes = self.eval_episodes
            
        if self.eval_verbose:
            print(f"Starting evaluation on {len(grids)} grids with {n_episodes} episodes each...")
        else:
            print(f"Evaluating on {n_episodes} episodes...")
        rewards = []
        
        for episode in range(n_episodes):
            if self.eval_verbose:
                print(f"  Episode {episode + 1}/{n_episodes}")
            
            # Randomly select a grid
            grid = random.choice(grids)
            if self.eval_verbose:
                print(f"    Selected grid with {np.count_nonzero(grid)} non-zero digits")
            
            try:
                env = RectangleEnv(initial_grid=grid)
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                step_count = 0
                
                while not done and step_count < 2000:  # Increased safety limit from 1000 to 2000
                    action, _ = self.model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = action.item()  # Convert numpy array to scalar
                    
                    obs, reward, done, _, _ = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    if self.eval_verbose and step_count % 100 == 0:
                        print(f"      Step {step_count}, reward so far: {episode_reward:.2f}")
                
                if step_count >= 2000:
                    if self.eval_verbose:
                        print(f"    WARNING: Episode stopped after {step_count} steps (safety limit)")
                
                if self.eval_verbose:
                    print(f"    Episode completed: {step_count} steps, total reward: {episode_reward:.2f}")
                rewards.append(episode_reward)
                
            except Exception as e:
                print(f"    ERROR in episode {episode + 1}: {str(e)}")
                rewards.append(0.0)  # Add zero reward for failed episode
        
        mean_reward = np.mean(rewards)
        if self.eval_verbose:
            print(f"Evaluation completed. Mean reward: {mean_reward:.2f}")
        return mean_reward


def create_training_plot(training_data, model_timestamp, save_dir="plots"):
    """
    Create and save training performance plots.
    
    Args:
        training_data: Dictionary containing training metrics
        model_timestamp: Timestamp of the model for naming the plot file
        save_dir: Directory to save the plot
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Performance - Model {model_timestamp}', fontsize=16)
    
    # Plot 1: Mean Reward over Time
    if training_data['timesteps'] and training_data['mean_rewards']:
        ax1.plot(training_data['timesteps'], training_data['mean_rewards'], 
                'b-', alpha=0.7, label='Training Reward')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Training Reward Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Plot 2: Evaluation Reward over Time
    if training_data['eval_timesteps'] and training_data['eval_mean_rewards']:
        ax2.plot(training_data['eval_timesteps'], training_data['eval_mean_rewards'], 
                'r-', marker='o', markersize=4, label='Evaluation Reward')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Evaluation Reward Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Plot 3: Episode Length over Time
    if training_data['timesteps'] and training_data['episode_lengths']:
        ax3.plot(training_data['timesteps'], training_data['episode_lengths'], 
                'g-', alpha=0.7, label='Episode Length')
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Mean Episode Length')
        ax3.set_title('Episode Length Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # Plot 4: Combined Training and Evaluation
    if (training_data['timesteps'] and training_data['mean_rewards'] and 
        training_data['eval_timesteps'] and training_data['eval_mean_rewards']):
        ax4.plot(training_data['timesteps'], training_data['mean_rewards'], 
                'b-', alpha=0.7, label='Training Reward')
        ax4.plot(training_data['eval_timesteps'], training_data['eval_mean_rewards'], 
                'r-', marker='o', markersize=4, label='Evaluation Reward')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Mean Reward')
        ax4.set_title('Training vs Evaluation Reward')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plot_filename = f"training_performance_{model_timestamp}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training performance plot saved to: {plot_path}")
    return plot_path


def save_training_data(training_data, model_timestamp, save_dir="training_data"):
    """
    Save training data to a file for later analysis.
    
    Args:
        training_data: Dictionary containing training metrics
        model_timestamp: Timestamp of the model for naming the file
        save_dir: Directory to save the data
    """
    os.makedirs(save_dir, exist_ok=True)
    data_filename = f"training_data_{model_timestamp}.pkl"
    data_path = os.path.join(save_dir, data_filename)
    
    with open(data_path, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"Training data saved to: {data_path}")
    return data_path


def load_training_data(model_timestamp, save_dir="training_data"):
    """
    Load training data from a file.
    
    Args:
        model_timestamp: Timestamp of the model
        save_dir: Directory containing the data file
        
    Returns:
        Dictionary containing training metrics
    """
    data_filename = f"training_data_{model_timestamp}.pkl"
    data_path = os.path.join(save_dir, data_filename)
    
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Training data loaded from: {data_path}")
        return training_data
    else:
        print(f"Training data file not found: {data_path}")
        return None


# --- 3. Main Training Script ---

# Example grid for training
EXAMPLE_GRID = np.array([
    [9, 7, 5, 5, 9, 8, 1, 3, 5, 9, 7, 6, 5, 2, 2, 4, 8],
    [8, 4, 3, 2, 4, 5, 4, 3, 8, 3, 5, 2, 6, 2, 5, 2, 7],
    [5, 2, 4, 7, 7, 4, 9, 2, 4, 1, 1, 8, 6, 8, 5, 3, 7],
    [6, 4, 7, 5, 6, 9, 2, 5, 9, 9, 7, 6, 6, 7, 4, 9, 3],
    [8, 4, 3, 5, 5, 9, 8, 4, 7, 8, 1, 5, 8, 5, 6, 6, 3],
    [4, 2, 5, 8, 2, 9, 7, 7, 9, 2, 6, 9, 3, 3, 5, 6, 5],
    [6, 9, 8, 5, 3, 8, 2, 5, 7, 2, 8, 6, 1, 7, 1, 4, 8],
    [2, 6, 6, 9, 5, 7, 8, 6, 5, 8, 8, 5, 4, 8, 7, 8, 1],
    [1, 1, 3, 8, 6, 8, 6, 4, 8, 9, 8, 8, 4, 5, 2, 1, 5],
    [2, 3, 1, 3, 3, 2, 3, 7, 8, 6, 2, 7, 9, 2, 9, 8, 3]
], dtype=np.uint8)

def make_env(rank, seed=0):
    """
    Utility function to create a single environment.
    Args:
        rank: The index of the subprocess
        seed: The seed for the environment
    """
    def _init():
        env = RectangleEnv(initial_grid=EXAMPLE_GRID, render_mode="none")
        env.reset(seed=seed + rank)
        return env
    return _init

def save_model(model, name=None):
    """Save the model with timestamp."""
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"rectangle_solver_ppo_{timestamp}"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", name)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """Load a saved model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return PPO.load(model_path)

def evaluate_model(model, env, num_episodes=5):
    """Evaluate the model's performance."""
    print("\nEvaluating model performance:")
    print("-" * 40)
    
    total_rewards = []
    total_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Episode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
    
    print("\nSummary:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Length: {np.mean(total_lengths):.2f} ± {np.std(total_lengths):.2f}")

def load_training_grids(data_dir: str = "training_grids") -> List[np.ndarray]:
    """
    Load all saved grids from the training_grids directory.
    Only includes grids with exactly 170 non-zero digits.
    
    Args:
        data_dir: Directory containing saved grids
        
    Returns:
        List of valid grids (numpy arrays)
    """
    grids = []
    invalid_grids = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} directory not found")
        return grids
        
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    grid = pickle.load(f)
                    
                # Count non-zero digits
                non_zero_count = np.count_nonzero(grid)
                
                if non_zero_count == 170:
                    grids.append(grid)
                else:
                    invalid_grids.append((filename, non_zero_count))
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if invalid_grids:
        print("\nInvalid grids found (not 170 non-zero digits):")
        for filename, count in invalid_grids:
            print(f"  {filename}: {count} non-zero digits")
    
    print(f"\nLoaded {len(grids)} valid grids")
    return grids

def prepare_training_data(grids: List[np.ndarray],
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split grids into training, validation, and test sets.
    Only includes grids with exactly 170 non-zero digits.
    
    Args:
        grids: List of valid grids (must have 170 non-zero digits)
        train_ratio: Proportion of grids to use for training
        val_ratio: Proportion of grids to use for validation
        
    Returns:
        Tuple of (train_grids, val_grids, test_grids)
    """
    if not grids:
        print("No valid grids available for training")
        return [], [], []
    
    # Verify all grids are valid
    valid_grids = []
    for grid in grids:
        if np.count_nonzero(grid) == 170:
            valid_grids.append(grid)
        else:
            print(f"Warning: Found grid with {np.count_nonzero(grid)} non-zero digits, skipping")
    
    if not valid_grids:
        print("No valid grids (170 non-zero digits) found in the dataset")
        return [], [], []
    
    # Shuffle the grids
    random.shuffle(valid_grids)
    
    # Calculate split indices
    n_grids = len(valid_grids)
    n_train = int(n_grids * train_ratio)
    n_val = int(n_grids * val_ratio)
    
    # Split the grids
    train_grids = valid_grids[:n_train]
    val_grids = valid_grids[n_train:n_train + n_val]
    test_grids = valid_grids[n_train + n_val:]
    
    print(f"\nSplit {n_grids} valid grids into:")
    print(f"  Training:   {len(train_grids)} grids")
    print(f"  Validation: {len(val_grids)} grids")
    print(f"  Test:       {len(test_grids)} grids")
    
    return train_grids, val_grids, test_grids

def analyze_grids(grids: List[np.ndarray]) -> None:
    """
    Print statistics about the collected grids.
    
    Args:
        grids: List of grids to analyze
    """
    if not grids:
        print("No grids to analyze")
        return
    
    # Calculate statistics
    n_grids = len(grids)
    shapes = [grid.shape for grid in grids]
    unique_shapes = set(shapes)
    
    # Count non-zero digits in each grid
    non_zero_counts = [np.count_nonzero(grid) for grid in grids]
    avg_non_zero = np.mean(non_zero_counts)
    std_non_zero = np.std(non_zero_counts)
    
    # Count digit frequencies
    digit_counts = {}
    for grid in grids:
        for digit in grid.flatten():
            if digit != 0:
                digit_counts[digit] = digit_counts.get(digit, 0) + 1
    
    # Print statistics
    print("\nGrid Analysis:")
    print(f"Total grids: {n_grids}")
    print(f"Grid shapes: {unique_shapes}")
    print(f"Average non-zero digits per grid: {avg_non_zero:.1f} ± {std_non_zero:.1f}")
    print("\nDigit frequencies:")
    for digit in sorted(digit_counts.keys()):
        count = digit_counts[digit]
        percentage = (count / sum(digit_counts.values())) * 100
        print(f"Digit {digit}: {count} occurrences ({percentage:.1f}%)")

def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():  # For Apple Silicon
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
        return device
    else:
        print("No GPU available, using CPU")
        return torch.device("cpu")

def train_model_with_saved_grids(model_path: Optional[str] = None,
                               data_dir: str = "data/train",  # Changed default to organized train directory
                               val_dir: str = "data/validation",  # Added validation directory parameter
                               total_timesteps: int = 1_000_000,
                               verbose: int = 1,
                               use_tensorboard: bool = False,
                               fast_mode: bool = False,
                               fast_eval: bool = False) -> None:
    """
    Train a model using organized data directories.
    
    Args:
        model_path: Path to existing model to continue training (None for new model)
        data_dir: Directory containing training grids (data/train)
        val_dir: Directory containing validation grids (data/validation)
        total_timesteps: Number of timesteps to train for
        verbose: Verbosity level for training
        use_tensorboard: Whether to enable tensorboard logging
        fast_mode: Whether to use faster training settings
        fast_eval: Whether to use faster evaluation settings
    """
    # Force CPU usage for Intel/AMD systems
    device = torch.device("cpu")
    print("Using CPU for training (optimized for Intel i9)")
    
    # Check if organized data directories exist
    if not os.path.exists(data_dir):
        print(f"❌ Training data directory not found: {data_dir}")
        print("Please run 'make organize-data' first to organize your data.")
        return
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation data directory not found: {val_dir}")
        print("Please run 'make organize-data' first to organize your data.")
        return
    
    # Load training grids
    train_grids = load_training_grids(data_dir)
    if not train_grids:
        print(f"No training data available in {data_dir}")
        return
    
    # Load validation grids
    val_grids = load_training_grids(val_dir)
    if not val_grids:
        print(f"No validation data available in {val_dir}")
        return
    
    print(f"✅ Loaded {len(train_grids)} training grids from {data_dir}")
    print(f"✅ Loaded {len(val_grids)} validation grids from {val_dir}")
    
    # Analyze the training grids
    analyze_grids(train_grids)
    
    # Create vectorized environment with training grids
    # Use fewer environments in fast mode
    n_envs = min(os.cpu_count() or 4, 4 if fast_mode else 8)
    print(f"Using {n_envs} parallel environments")
    env = make_vec_env(
        RectangleEnv,
        n_envs=n_envs,
        env_kwargs={'grids': train_grids},
        monitor_dir='logs'
    )
    
    # Adjust training parameters based on mode
    if fast_mode:
        print("Using fast training mode with optimized parameters")
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 5e-4,  # Slightly higher learning rate
            "n_steps": 4096,  # More steps per update
            "batch_size": 512,  # Larger batches
            "n_epochs": 8,  # Fewer epochs
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
            "device": device,
            "policy_kwargs": dict(
                net_arch=dict(
                    pi=[128, 128],  # Smaller network for faster training
                    vf=[128, 128]   # Smaller network for faster training
                ),
                activation_fn=torch.nn.ReLU
            )
        }
    else:
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
            "device": device,
            "policy_kwargs": dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                ),
                activation_fn=torch.nn.ReLU
            )
        }

    # Add tensorboard logging if enabled
    if use_tensorboard:
        try:
            import tensorboard
            model_kwargs["tensorboard_log"] = "./logs/"
            print("Tensorboard logging enabled")
        except ImportError:
            print("Warning: Tensorboard not installed, disabling logging")
            use_tensorboard = False
    
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("Creating new model")
        model = PPO(**model_kwargs)
    
    # Train the model
    print(f"\nTraining for {total_timesteps} timesteps...")
    
    # Configure evaluation settings based on fast_eval
    if fast_eval:
        eval_freq = 50000  # Evaluate every 50k steps instead of 20k (much less frequent)
        eval_episodes = 1   # Only 1 episode per evaluation (faster)
        eval_verbose = False  # Minimal output
        print("Using fast evaluation mode:")
        print(f"  - Evaluation frequency: every {eval_freq:,} steps")
        print(f"  - Episodes per evaluation: {eval_episodes}")
        print(f"  - Verbose output: {eval_verbose}")
    else:
        eval_freq = 10000   # Default: every 10k steps
        eval_episodes = 3   # Default: 3 episodes
        eval_verbose = True # Default: verbose output
    
    # Create callback
    callback = CustomCallback(
        val_grids=val_grids,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        eval_verbose=eval_verbose
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/model_{timestamp}.zip"
    model.save(save_path)
    print(f"\nSaved model to {save_path}")

    # Create training plot
    create_training_plot(callback.training_data, timestamp)
    
    # Save training data
    save_training_data(callback.training_data, timestamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model using saved grids')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'solve', 'continue'], default='train',
                      help='Mode to run the script in')
    parser.add_argument('--model', help='Path to existing model to continue training')
    parser.add_argument('--data-dir', default='data/train', help='Directory containing training grids (default: data/train)')
    parser.add_argument('--val-dir', default='data/validation', help='Directory containing validation grids (default: data/validation)')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Number of timesteps to train for')
    parser.add_argument('--use-saved-grids', action='store_true', help='Use organized data directories for training')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze grids without training')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes for evaluation')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering during evaluation/solving')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--tensorboard', action='store_true', help='Enable tensorboard logging')
    parser.add_argument('--fast', action='store_true', help='Use fast training mode with optimized parameters')
    parser.add_argument('--fast-eval', action='store_true', help='Use fast evaluation mode (less frequent, fewer episodes)')
    parser.add_argument('--create-plot', help='Create training plot for a specific model timestamp')
    args = parser.parse_args()
    
    # Always use CPU for Intel/AMD systems
    device = torch.device("cpu")
    print("Using CPU for training (optimized for Intel i9)")
    
    if args.create_plot:
        # Create plot for existing model
        training_data = load_training_data(args.create_plot)
        if training_data:
            create_training_plot(training_data, args.create_plot)
        else:
            print(f"Could not load training data for model {args.create_plot}")

    if args.use_saved_grids:
        # Use organized data directories for training
        train_model_with_saved_grids(
            model_path=args.model,
            data_dir=args.data_dir,
            val_dir=args.val_dir,
            total_timesteps=args.timesteps,
            use_tensorboard=args.tensorboard,
            fast_mode=args.fast,
            fast_eval=args.fast_eval
        )
    elif args.analyze_only:
        # Only analyze the grids
        grids = load_training_grids(args.data_dir)
        analyze_grids(grids)
    elif args.mode == 'train':
        # Original training with random grids
        n_envs = min(os.cpu_count() or 4, 8)  # Use up to 8 cores
        print(f"Using {n_envs} parallel environments")
        env = make_vec_env(
            RectangleEnv,
            n_envs=n_envs,
            monitor_dir='logs'
        )
        
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 3e-4,
            "n_steps": 2048,  # Increased for better CPU utilization
            "batch_size": 256,  # Increased for better CPU utilization
            "n_epochs": 10,  # Standard value for CPU
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
            "device": device,
            "policy_kwargs": dict(
                net_arch=dict(
                    pi=[256, 256],  # Larger network for better learning
                    vf=[256, 256]   # Larger network for better learning
                ),
                activation_fn=torch.nn.ReLU
            )
        }
        
        # Add tensorboard logging if enabled
        if args.tensorboard:
            try:
                import tensorboard
                model_kwargs["tensorboard_log"] = "./logs/"
                print("Tensorboard logging enabled")
            except ImportError:
                print("Warning: Tensorboard not installed, disabling logging")
                args.tensorboard = False
        
        if args.model and os.path.exists(args.model):
            print(f"Loading existing model from {args.model}")
            model = PPO.load(args.model, env=env, device=device)
        else:
            print("Creating new model")
            model = PPO(**model_kwargs)
        
        model.learn(
            total_timesteps=args.timesteps,
            callback=CustomCallback()
        )
        
        # Save the model
        save_model(model)
        
    elif args.mode == 'evaluate':
        if not args.model:
            print("Error: Please specify a model path using --model")
            exit(1)
            
        env = RectangleEnv(render_mode="human" if not args.no_render else None)
        model = load_model(args.model, device=device)
        evaluate_model(model, env, num_episodes=args.eval_episodes)
        
    elif args.mode == 'solve':
        if not args.model:
            print("Error: Please specify a model path using --model")
            exit(1)
            
        env = RectangleEnv(render_mode="human" if not args.no_render else None)
        model = load_model(args.model, device=device)
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
        print(f"\nTotal reward: {total_reward}")
        print(f"Digits cleared: {env.episode_digits_cleared}")
        
    elif args.mode == 'continue':
        if not args.model:
            print("Error: Please specify a model path using --model")
            exit(1)
            
        env = make_vec_env(
            RectangleEnv,
            n_envs=4,
            monitor_dir='logs'
        )
        
        model = PPO.load(args.model, env=env, device=device)
        model.learn(
            total_timesteps=args.timesteps,
            callback=CustomCallback()
        )
        
        # Save the model
        save_model(model)
