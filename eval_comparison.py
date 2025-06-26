#!/usr/bin/env python3
"""
Evaluation Mode Comparison Script

This script demonstrates the trade-offs between different evaluation settings
for the rectangle solver training.
"""

import time
import numpy as np
from training import RectangleEnv, CustomCallback

def simulate_evaluation_time(eval_freq, eval_episodes, total_steps=100000):
    """
    Simulate the time spent on evaluation with different settings.
    
    Args:
        eval_freq: How often to evaluate (steps)
        eval_episodes: Number of episodes per evaluation
        total_steps: Total training steps to simulate
        
    Returns:
        dict with timing information
    """
    # Estimate time per episode (based on typical performance)
    time_per_episode = 2.0  # seconds (typical for a 10x17 grid)
    
    # Calculate number of evaluations
    num_evaluations = total_steps // eval_freq
    
    # Calculate total evaluation time
    total_eval_time = num_evaluations * eval_episodes * time_per_episode
    
    # Calculate percentage of time spent on evaluation
    # Assuming training runs at ~1000 steps/second
    training_time = total_steps / 1000  # seconds
    eval_percentage = (total_eval_time / (training_time + total_eval_time)) * 100
    
    return {
        'num_evaluations': num_evaluations,
        'total_eval_time': total_eval_time,
        'training_time': training_time,
        'total_time': training_time + total_eval_time,
        'eval_percentage': eval_percentage,
        'time_per_evaluation': eval_episodes * time_per_episode
    }

def print_comparison():
    """Print comparison of different evaluation settings."""
    
    print("Evaluation Mode Comparison")
    print("=" * 50)
    print()
    
    # Define different evaluation modes
    modes = {
        'Original': {'freq': 5000, 'episodes': 5},
        'Default': {'freq': 10000, 'episodes': 3},
        'Fast': {'freq': 20000, 'episodes': 2},
        'Minimal': {'freq': 50000, 'episodes': 1},
    }
    
    total_steps = 1000000  # 1M steps
    
    print(f"Simulating {total_steps:,} training steps...")
    print()
    
    for mode_name, settings in modes.items():
        results = simulate_evaluation_time(
            settings['freq'], 
            settings['episodes'], 
            total_steps
        )
        
        print(f"{mode_name} Mode:")
        print(f"  - Evaluation frequency: every {settings['freq']:,} steps")
        print(f"  - Episodes per evaluation: {settings['episodes']}")
        print(f"  - Number of evaluations: {results['num_evaluations']}")
        print(f"  - Time per evaluation: {results['time_per_evaluation']:.1f}s")
        print(f"  - Total evaluation time: {results['total_eval_time']/60:.1f} minutes")
        print(f"  - Total training time: {results['training_time']/60:.1f} minutes")
        print(f"  - Total time: {results['total_time']/60:.1f} minutes")
        print(f"  - Time spent on evaluation: {results['eval_percentage']:.1f}%")
        print()

def print_trade_offs():
    """Print the trade-offs of different evaluation settings."""
    
    print("Trade-offs Analysis")
    print("=" * 50)
    print()
    
    print("Benefits of Less Evaluation:")
    print("✓ Faster overall training time")
    print("✓ Less computational overhead")
    print("✓ More time spent on actual learning")
    print("✓ Lower memory usage during evaluation")
    print()
    
    print("Downsides of Less Evaluation:")
    print("✗ Less reliable model selection (might save suboptimal models)")
    print("✗ Harder to detect overfitting early")
    print("✗ Less confidence in convergence")
    print("✗ May miss optimal stopping point")
    print("✗ Less detailed progress tracking")
    print()
    
    print("Recommendations:")
    print("• For initial exploration: Use Fast or Minimal mode")
    print("• For final training: Use Default mode")
    print("• For production: Use Default mode with periodic full evaluations")
    print()

if __name__ == "__main__":
    print_comparison()
    print_trade_offs()
    
    print("Usage Examples:")
    print("=" * 50)
    print()
    print("# Fast training with minimal evaluation:")
    print("python training.py --use-saved-grids --fast --fast-eval")
    print()
    print("# Standard training with default evaluation:")
    print("python training.py --use-saved-grids")
    print()
    print("# Thorough training with frequent evaluation:")
    print("python training.py --use-saved-grids --eval-episodes 5")
    print() 