#!/usr/bin/env python3
"""
Script to evaluate a trained model on training grids and show detailed results.
"""

import os
import glob
import pickle
import numpy as np
from stable_baselines3 import PPO
from training import RectangleEnv
import argparse

def get_latest_model():
    """Get the path to the most recent model file."""
    model_files = glob.glob("models/*.zip")
    if not model_files:
        raise FileNotFoundError("No model files found in models/ directory")
    
    # Sort by modification time (newest first)
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def get_random_training_grid():
    """Load a random training grid."""
    grid_files = glob.glob("training_grids/*.pkl")
    if not grid_files:
        raise FileNotFoundError("No training grid files found in training_grids/ directory")
    
    # Pick a random grid file
    import random
    random_grid_file = random.choice(grid_files)
    
    with open(random_grid_file, 'rb') as f:
        grid = pickle.load(f)
    
    return grid, os.path.basename(random_grid_file)

def evaluate_model_on_grid(model_path, grid, grid_name=None):
    """Evaluate a model on a specific grid and return detailed results."""
    # Load the model
    model = PPO.load(model_path)
    
    # Create environment with the specific grid
    env = RectangleEnv(initial_grid=grid, render_mode=None)
    
    # Reset environment
    obs, info = env.reset()
    
    # Track episode details
    episode_reward = 0
    episode_length = 0
    total_digits_cleared = 0
    initial_digits = env.initial_digits
    actions_taken = []
    previous_digits_cleared = 0  # Track digits cleared before this step
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {os.path.basename(model_path)}")
    print(f"GRID: {grid_name or 'Random training grid'}")
    print(f"INITIAL DIGITS: {initial_digits}")
    print(f"GRID SHAPE: {grid.shape}")
    print(f"{'='*60}")
    
    # Run the episode
    done = False
    step = 0
    
    while not done:
        # Get action from model - filter valid actions
        action, _ = model.predict(obs, deterministic=True)
        
        # Check if action is valid, if not pick a random valid action
        valid_actions = info["action_mask"]
        if not valid_actions[action]:
            valid_action_indices = np.where(valid_actions)[0]
            if len(valid_action_indices) > 0:
                action = np.random.choice(valid_action_indices)
            else:
                print("No valid actions available!")
                break
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        # Track results
        episode_reward += reward
        episode_length += 1
        
        # Get rectangle coordinates
        r1, c1, r2, c2 = env._get_rectangle_from_action(action)
        rect_size = (r2 - r1 + 1) * (c2 - c1 + 1)
        
        # Calculate digits cleared in this step correctly
        current_digits_cleared = env.episode_digits_cleared
        digits_cleared_this_step = current_digits_cleared - previous_digits_cleared
        previous_digits_cleared = current_digits_cleared
        
        # Handle None case for grid
        remaining_digits = np.count_nonzero(env.grid) if env.grid is not None else 0
        
        actions_taken.append({
            'step': step + 1,
            'action': action,
            'rectangle': (r1, c1, r2, c2),
            'rect_size': rect_size,
            'reward': reward,
            'digits_cleared': digits_cleared_this_step,
            'cumulative_reward': episode_reward,
            'remaining_digits': remaining_digits
        })
        
        step += 1
    
    # Final results
    final_digits_remaining = np.count_nonzero(env.grid) if env.grid is not None else 0
    digits_cleared_total = initial_digits - final_digits_remaining
    completion_percentage = (digits_cleared_total / initial_digits) * 100 if initial_digits > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Episode Length: {episode_length} actions")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Initial Digits: {initial_digits}")
    print(f"Digits Cleared: {digits_cleared_total}")
    print(f"Digits Remaining: {final_digits_remaining}")
    print(f"Completion: {completion_percentage:.1f}%")
    print(f"Average Reward per Action: {episode_reward/episode_length:.2f}")
    print(f"Average Digits per Action: {digits_cleared_total/episode_length:.2f}")
    
    if completion_percentage == 100:
        print(f"🎉 GRID COMPLETED! 🎉")
    
    print(f"\n{'='*60}")
    print(f"ACTION DETAILS:")
    print(f"{'='*60}")
    for action_info in actions_taken:
        print(f"Step {action_info['step']:2d}: "
              f"Rect({action_info['rectangle'][0]},{action_info['rectangle'][1]})->"
              f"({action_info['rectangle'][2]},{action_info['rectangle'][3]}) "
              f"Size:{action_info['rect_size']:2d} "
              f"Reward:{action_info['reward']:6.2f} "
              f"Digits:{action_info['digits_cleared']:3.0f} "
              f"Remaining:{action_info['remaining_digits']:3d}")
    
    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'total_digits_cleared': digits_cleared_total,
        'initial_digits': initial_digits,
        'completion_percentage': completion_percentage,
        'actions_taken': actions_taken
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on training grids')
    parser.add_argument('--model', type=str, help='Path to model file (default: latest)')
    parser.add_argument('--grid', type=str, help='Path to specific grid file (default: random)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run (default: 1)')
    
    args = parser.parse_args()
    
    try:
        # Get model path
        if args.model:
            model_path = args.model
        else:
            model_path = get_latest_model()
            print(f"Using latest model: {model_path}")
        
        # Run evaluation
        results = []
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            # Get grid
            if args.grid:
                with open(args.grid, 'rb') as f:
                    grid = pickle.load(f)
                grid_name = os.path.basename(args.grid)
            else:
                grid, grid_name = get_random_training_grid()
            
            # Evaluate
            result = evaluate_model_on_grid(model_path, grid, grid_name)
            results.append(result)
        
        # Summary if multiple episodes
        if args.episodes > 1:
            print(f"\n{'='*60}")
            print(f"SUMMARY ({args.episodes} episodes):")
            print(f"{'='*60}")
            avg_reward = np.mean([r['episode_reward'] for r in results])
            avg_length = np.mean([r['episode_length'] for r in results])
            avg_digits_cleared = np.mean([r['total_digits_cleared'] for r in results])
            avg_completion = np.mean([r['completion_percentage'] for r in results])
            
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Length: {avg_length:.1f}")
            print(f"Average Digits Cleared: {avg_digits_cleared:.1f}")
            print(f"Average Completion: {avg_completion:.1f}%")
            
            completed_episodes = sum(1 for r in results if r['completion_percentage'] == 100)
            print(f"Completed Episodes: {completed_episodes}/{args.episodes}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 