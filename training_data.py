import numpy as np
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pickle

class TrainingDataCollector:
    def __init__(self, data_dir: str = "training_data"):
        """
        Initialize the training data collector.
        
        Args:
            data_dir: Directory to store training data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create a new episode file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_file = os.path.join(data_dir, f"episode_{timestamp}.pkl")
        self.current_episode = {
            'initial_grid': None,
            'actions': [],  # List of (r1, c1, r2, c2) tuples
            'rewards': [],  # List of rewards for each action
            'total_reward': 0,
            'digits_cleared': 0,
            'num_actions': 0,
            'success': False,  # Whether the episode was successful
            'timestamp': timestamp
        }
    
    def start_episode(self, initial_grid: np.ndarray):
        """Start a new episode with the given initial grid."""
        self.current_episode = {
            'initial_grid': initial_grid.copy(),
            'actions': [],
            'rewards': [],
            'total_reward': 0,
            'digits_cleared': 0,
            'num_actions': 0,
            'success': False,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def record_action(self, action: Tuple[int, int, int, int], reward: float, 
                     digits_cleared: int, grid_after_action: np.ndarray):
        """
        Record an action taken in the current episode.
        
        Args:
            action: Tuple of (r1, c1, r2, c2) representing the rectangle
            reward: Reward received for this action
            digits_cleared: Number of digits cleared by this action
            grid_after_action: Grid state after the action
        """
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['total_reward'] += reward
        self.current_episode['digits_cleared'] = digits_cleared
        self.current_episode['num_actions'] += 1
        self.current_episode['grid_after_action'] = grid_after_action.copy()
    
    def end_episode(self, success: bool):
        """
        End the current episode and save it to disk.
        
        Args:
            success: Whether the episode was successful
        """
        self.current_episode['success'] = success
        
        # Save episode to file
        with open(self.episode_file, 'wb') as f:
            pickle.dump(self.current_episode, f)
        
        print(f"Saved episode data to {self.episode_file}")
        print(f"Episode summary:")
        print(f"- Actions taken: {self.current_episode['num_actions']}")
        print(f"- Total reward: {self.current_episode['total_reward']}")
        print(f"- Digits cleared: {self.current_episode['digits_cleared']}")
        print(f"- Success: {success}")

def collect_training_data_from_model(grid: np.ndarray, model_path: str, 
                                   collector: TrainingDataCollector) -> None:
    """
    Use a trained model to solve a grid and collect training data.
    
    Args:
        grid: Initial grid to solve
        model_path: Path to the trained model
        collector: TrainingDataCollector instance
    """
    from solver import solve_rectangles_model
    
    # Start new episode
    collector.start_episode(grid)
    
    try:
        # Use the model to solve the grid
        rectangles, final_grid, num_digits_cleared = solve_rectangles_model(grid, model_path)
        
        # Record each action
        current_grid = grid.copy()
        for i, (r1, c1, r2, c2) in enumerate(rectangles):
            # Calculate reward (area of rectangle)
            reward = (r2 - r1 + 1) * (c2 - c1 + 1)
            
            # Update grid
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    current_grid[r, c] = 0
            
            # Record the action
            collector.record_action(
                action=(r1, c1, r2, c2),
                reward=reward,
                digits_cleared=np.count_nonzero(grid) - np.count_nonzero(current_grid),
                grid_after_action=current_grid.copy()
            )
        
        # End episode
        success = num_digits_cleared > 0
        collector.end_episode(success)
        
    except Exception as e:
        print(f"Error during data collection: {str(e)}")
        collector.end_episode(False)
        raise

def collect_training_data_from_human(grid: np.ndarray, 
                                   collector: TrainingDataCollector) -> None:
    """
    Collect training data from human gameplay.
    
    Args:
        grid: Initial grid to solve
        collector: TrainingDataCollector instance
    """
    from solver import print_grid
    
    collector.start_episode(grid)
    current_grid = grid.copy()
    
    try:
        while True:
            print("\nCurrent grid:")
            print_grid(current_grid)
            
            # Get rectangle coordinates from user
            try:
                coords = input("Enter rectangle coordinates (r1 c1 r2 c2) or 'done' to finish: ")
                if coords.lower() == 'done':
                    break
                    
                r1, c1, r2, c2 = map(int, coords.split())
                
                # Validate rectangle
                if not (0 <= r1 <= r2 < grid.shape[0] and 0 <= c1 <= c2 < grid.shape[1]):
                    print("Invalid coordinates!")
                    continue
                
                # Calculate reward and update grid
                reward = (r2 - r1 + 1) * (c2 - c1 + 1)
                old_grid = current_grid.copy()
                
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        current_grid[r, c] = 0
                
                # Record the action
                collector.record_action(
                    action=(r1, c1, r2, c2),
                    reward=reward,
                    digits_cleared=np.count_nonzero(old_grid) - np.count_nonzero(current_grid),
                    grid_after_action=current_grid.copy()
                )
                
            except ValueError:
                print("Invalid input! Please enter four numbers or 'done'")
                continue
        
        # End episode
        success = np.count_nonzero(current_grid) < np.count_nonzero(grid)
        collector.end_episode(success)
        
    except KeyboardInterrupt:
        print("\nGameplay interrupted")
        collector.end_episode(False)
    except Exception as e:
        print(f"Error during data collection: {str(e)}")
        collector.end_episode(False)
        raise

def save_grid(grid: np.ndarray, data_dir: str = "training_grids", timestamp: str = None):
    """
    Save a grid to the training grids directory.
    
    Args:
        grid: The grid to save
        data_dir: Directory to store grids
        timestamp: Optional timestamp to use in filename (e.g. from screenshot)
    Returns:
        bool: True if grid was saved, False if file already existed
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = os.path.join(data_dir, f"grid_{timestamp}.pkl")
    
    # Check if file already exists
    if os.path.exists(filename):
        print(f"Grid file already exists: {filename}")
        return False
    
    with open(filename, 'wb') as f:
        pickle.dump(grid, f)
    print(f"Saved grid to {filename}")
    return True

def process_screenshots_to_grids(screenshots_dir: str = "screenshots", 
                               training_grids_dir: str = "training_grids",
                               method: str = 'pixel',
                               verbose: bool = False) -> int:
    """
    Process all screenshots in the screenshots directory and save extracted grids.
    
    Args:
        screenshots_dir: Directory containing screenshot images
        training_grids_dir: Directory to save extracted grids
        method: Method to use for digit recognition ('template', 'ocr', or 'pixel')
        verbose: Whether to print verbose output
        
    Returns:
        int: Number of grids successfully processed and saved
    """
    from ocr_grid import process_image_to_grid, print_grid
    import os
    from glob import glob
    import re
    
    # Create training grids directory if it doesn't exist
    os.makedirs(training_grids_dir, exist_ok=True)
    
    # Get all PNG files in screenshots directory
    screenshot_files = sorted(glob(os.path.join(screenshots_dir, "*.png")))
    if not screenshot_files:
        print(f"No PNG files found in {screenshots_dir}")
        return 0
    
    print(f"Found {len(screenshot_files)} screenshots to process")
    successful_grids = 0
    skipped_grids = 0
    
    # Pattern to extract timestamp from screenshot filename
    timestamp_pattern = re.compile(r'screenshot_(\d{8}_\d{6})\.png')
    
    for screenshot_path in screenshot_files:
        try:
            screenshot_name = os.path.basename(screenshot_path)
            print(f"\nProcessing {screenshot_name}...")
            
            # Extract timestamp from screenshot filename
            match = timestamp_pattern.match(screenshot_name)
            if not match:
                print(f"Skipping {screenshot_name} - invalid filename format")
                continue
                
            timestamp = match.group(1)
            
            # Check if grid already exists
            grid_path = os.path.join(training_grids_dir, f"grid_{timestamp}.pkl")
            if os.path.exists(grid_path):
                print(f"Grid already exists for {screenshot_name}, skipping...")
                skipped_grids += 1
                continue
            
            # Process the image to get grid
            grid, results = process_image_to_grid(
                image_path=screenshot_path,
                method=method,
                verbose=verbose
            )
            
            # Print grid for verification
            if verbose:
                print("Extracted grid:")
                print_grid(grid)
            
            # Save grid using existing save_grid function
            if save_grid(grid, data_dir=training_grids_dir, timestamp=timestamp):
                successful_grids += 1
                print(f"Successfully processed {screenshot_name}")
            
        except Exception as e:
            print(f"Error processing {screenshot_name}: {str(e)}")
            continue
    
    print(f"\nProcessing complete:")
    print(f"- Successfully saved {successful_grids} new grids")
    print(f"- Skipped {skipped_grids} existing grids")
    print(f"- Total grids in {training_grids_dir}: {successful_grids + skipped_grids}")
    return successful_grids

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description='Process screenshots to training grids')
    parser.add_argument('--screenshots-dir', default='screenshots', help='Directory containing screenshots')
    parser.add_argument('--training-grids-dir', default='training_grids', help='Directory to save extracted grids')
    parser.add_argument('--method', default='pixel', choices=['template', 'ocr', 'pixel'], help='Method for digit recognition')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    process_screenshots_to_grids(
        screenshots_dir=args.screenshots_dir,
        training_grids_dir=args.training_grids_dir,
        method=args.method,
        verbose=args.verbose
    ) 