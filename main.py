from ocr_grid import process_image_to_grid
from ocr_grid import print_grid
from solver import find_rectangles_zero_inclusive_greedy
from solver import find_rectangles_digit_priority_greedy
from solver import solve_rectangles_backtracking
from solver import solve_rectangles_model
import time
import numpy as np
from clicker import process_and_click, ScreenOffset
from typing import Optional, Tuple, Union
import pyautogui
from utils import print_verbose
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import json
import glob

test_image = "test_data/test_4.png"

RESET_LOC = (55,1011)
PLAY_LOC = (320,613)

# Global data collection - will be loaded per model
attempt_data = {
    'attempts': [],
    'scores': [],
    'timestamps': [],
    'grids_saved': [],
    'threshold_met': []
}

def get_model_name(model_path: str) -> str:
    """Extract model name from path for data organization."""
    # Extract filename without extension
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return model_name

def load_model_data(model_name: str) -> dict:
    """Load existing performance data for a specific model."""
    data_file = f"performance_data_{model_name}.json"
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded existing data for model {model_name}: {len(data['attempts'])} attempts")
                return data
        except Exception as e:
            print(f"Error loading data for {model_name}: {e}")
    
    # Return empty data structure if no existing data
    return {
        'attempts': [],
        'scores': [],
        'timestamps': [],
        'grids_saved': [],
        'threshold_met': []
    }

def save_attempt_data(model_name: str):
    """Save just the attempt data without creating a plot (for frequent saves)."""
    data_file = f"performance_data_{model_name}.json"
    
    # Convert NumPy types to Python types for JSON serialization
    json_safe_data = {
        'attempts': [int(x) for x in attempt_data['attempts']],
        'scores': [int(x) for x in attempt_data['scores']],
        'timestamps': attempt_data['timestamps'],
        'grids_saved': attempt_data['grids_saved'],
        'threshold_met': [bool(x) for x in attempt_data['threshold_met']]
    }
    
    with open(data_file, 'w') as f:
        json.dump(json_safe_data, f, indent=2)

def save_performance_data(model_name: str):
    """Save the collected performance data to files for a specific model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data
    data_file = f"performance_data_{model_name}.json"
    
    # Convert NumPy types to Python types for JSON serialization
    json_safe_data = {
        'attempts': [int(x) for x in attempt_data['attempts']],
        'scores': [int(x) for x in attempt_data['scores']],
        'timestamps': attempt_data['timestamps'],
        'grids_saved': attempt_data['grids_saved'],
        'threshold_met': [bool(x) for x in attempt_data['threshold_met']]
    }
    
    with open(data_file, 'w') as f:
        json.dump(json_safe_data, f, indent=2)
    
    # Create and save plot
    create_performance_plot(model_name, timestamp)
    
    print(f"Performance data saved to {data_file}")

def create_performance_plot(model_name: str, timestamp: Optional[str] = None):
    """Create a plot showing performance over all attempts for a specific model."""
    if len(attempt_data['attempts']) < 2:
        print("Not enough data to create plot")
        return
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Performance Over Time - {model_name} ({timestamp})', fontsize=16)
    
    # Plot 1: Score over attempts
    ax1.plot(attempt_data['attempts'], attempt_data['scores'], 'b-o', alpha=0.7, markersize=4)
    ax1.axhline(y=150, color='r', linestyle='--', alpha=0.7, label='Threshold (150)')
    ax1.set_xlabel('Attempt Number')
    ax1.set_ylabel('Digits Cleared')
    ax1.set_title('Performance Over Attempts')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Score distribution
    ax2.hist(attempt_data['scores'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=150, color='r', linestyle='--', alpha=0.7, label='Threshold (150)')
    ax2.set_xlabel('Digits Cleared')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    ax2.legend()
    
    # Plot 3: Success rate over time
    if len(attempt_data['attempts']) > 10:
        window_size = min(10, len(attempt_data['attempts']) // 4)
        success_rates = []
        attempt_windows = []
        
        for i in range(window_size, len(attempt_data['attempts'])):
            window = attempt_data['threshold_met'][i-window_size:i]
            success_rate = sum(window) / len(window)
            success_rates.append(success_rate)
            attempt_windows.append(i)
        
        ax3.plot(attempt_windows, success_rates, 'g-o', alpha=0.7, markersize=4)
        ax3.set_xlabel('Attempt Number')
        ax3.set_ylabel('Success Rate (rolling average)')
        ax3.set_title(f'Success Rate (window={window_size})')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # Plot 4: Cumulative statistics
    cumulative_success = np.cumsum(attempt_data['threshold_met'])
    cumulative_attempts = np.arange(1, len(attempt_data['attempts']) + 1)
    overall_success_rate = cumulative_success / cumulative_attempts
    
    ax4.plot(attempt_data['attempts'], overall_success_rate, 'purple', alpha=0.7, linewidth=2)
    ax4.set_xlabel('Attempt Number')
    ax4.set_ylabel('Overall Success Rate')
    ax4.set_title('Cumulative Success Rate')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"performance_plot_{model_name}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to {plot_file}")
    
    # Print summary statistics
    print(f"\nPerformance Summary for {model_name}:")
    print(f"Total attempts: {len(attempt_data['attempts'])}")
    print(f"Successful attempts: {sum(attempt_data['threshold_met'])}")
    print(f"Overall success rate: {sum(attempt_data['threshold_met'])/len(attempt_data['attempts'])*100:.1f}%")
    print(f"Average score: {np.mean(attempt_data['scores']):.1f}")
    print(f"Best score: {max(attempt_data['scores'])}")
    print(f"Worst score: {min(attempt_data['scores'])}")

def save_grid(grid: np.ndarray, data_dir: str = "training_grids"):
    """
    Save a grid to the training grids directory.
    
    Args:
        grid: The grid to save
        data_dir: Directory to store grids
    """
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(data_dir, f"grid_{timestamp}.pkl")
    
    with open(filename, 'wb') as f:
        pickle.dump(grid, f)
    print(f"Saved grid to {filename}")
    return filename

def process_and_solve(region: Optional[Tuple[int, int, int, int]] = None,
                     screen_offset: Optional[Union[ScreenOffset, Tuple[int, int]]] = None,
                     method: str = 'pixel',
                     duration: float = 30.0,
                     preview: bool = True,
                     verbose: bool = False,
                     use_test_image: bool = False,
                     model_path: str = "models/model_20250626_102548.zip") -> bool:
    """
    Process the current screen state, solve the grid, and click the solution.
    
    Args:
        region: Optional (left, top, width, height) tuple to capture a specific region
        screen_offset: Either a ScreenOffset object, (x,y) tuple, or None
        method: Method to use for digit recognition ('template', 'ocr', or 'pixel')
        duration: Total time to spend clicking (in seconds)
        preview: Whether to show a preview of the rectangles before clicking
        verbose: Whether to print verbose output
        use_test_image: Whether to use the test image instead of the current screen
        model_path: Path to the model to use for solving
    Returns:
        bool: True if clicking was completed, False if aborted
    """
    global attempt_data
    
    # Load existing data for this model
    model_name = get_model_name(model_path)
    attempt_data = load_model_data(model_name)
    
    timer = time.time()
    attempt_num = len(attempt_data['attempts']) + 1
    
    print(f"\n{'='*60}")
    print(f"ATTEMPT #{attempt_num} - Model: {model_name}")
    print(f"{'='*60}")
    
    # Process the image and get grid
    print_verbose("Processing image...", verbose)
    image_path = test_image if use_test_image else None
    grid, results = process_image_to_grid(image_path, method, screenshot_region=region, verbose=verbose)
    print_verbose("Extracted grid:", verbose)
    print_grid(grid)
    
    # Save the grid for later training
    grid_file = save_grid(grid)
    
    # Find solution
    print_verbose("Finding solution...", verbose)
    # solutions, final_grid, num_digits_cleared = solve_rectangles_backtracking(grid)
    solutions, final_grid, num_digits_cleared = solve_rectangles_model(grid, model_path)
    print_verbose("Found rectangles:", verbose)
    for r1, c1, r2, c2 in solutions:
        print_verbose(f"({r1},{c1}) to ({r2},{c2})", verbose)
    
    elapsed_time = round(time.time() - timer, 3)
    print_verbose(f"\nTime taken to process and solve: {elapsed_time} seconds", verbose)
    print(f"Number of apples found: {num_digits_cleared}") # Always printed

    # Collect data for this attempt
    attempt_data['attempts'].append(attempt_num)
    attempt_data['scores'].append(num_digits_cleared)
    attempt_data['timestamps'].append(datetime.now().isoformat())
    attempt_data['grids_saved'].append(grid_file)
    attempt_data['threshold_met'].append(num_digits_cleared >= 150)
    
    # Save data immediately after each attempt to prevent data loss
    save_attempt_data(model_name)
    
    # Create plot after every 5 attempts
    if attempt_num % 5 == 0:
        create_performance_plot(model_name, "live")
    
    if num_digits_cleared < 150:
        print(f"Not enough apples, only found {num_digits_cleared} apples. Resting...")
        reset_game()
        return process_and_solve(
            region=region,
            screen_offset=screen_offset,
            method=method,
            duration=duration,
            preview=preview,
            verbose=verbose,
            use_test_image=use_test_image,
            model_path=model_path
        )
    
    # Click the solution
    print_verbose("\nClicking solution...", verbose)
    success = process_and_click(
        image_path=image_path,  # Will take a new screenshot
        solution=solutions,  # type: ignore
        screen_offset=screen_offset,
        method=method,
        duration=duration,
        preview=preview,
        screenshot_region=region,
        results=results,
        verbose=verbose
    )
    
    return success

def reset_game():
    """Reset the game by clicking the play button."""
    pyautogui.click(x=RESET_LOC[0], y=RESET_LOC[1], clicks=2) # click twice to make sure the window is focused
    time.sleep(1)
    pyautogui.click(x=PLAY_LOC[0], y=PLAY_LOC[1], clicks=1)

def main():
    """Example usage of the process_and_solve function."""
    # Example coordinates for the game window
    # These should be adjusted based on your screen setup
    region = (18, 202, 1223, 723)  # (left, top, width, height)
    screen_offset = (18, 202)     # (x, y) offset

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Use test image instead of screenshot')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--max-attempts', type=int, default=None, help='Maximum number of attempts before stopping')
    parser.add_argument('--model', type=str, default="models/model_20250626_102548.zip", help='Path to model to use')
    args = parser.parse_args()
    
    try:
        # Process and solve
        success = process_and_solve(
            region=region,
            screen_offset=screen_offset,
            method='pixel',
            duration=120.0,
            preview=True,
            verbose=args.verbose,
            use_test_image=args.test,
            model_path=args.model
        )
        
        if success:
            print("\nSuccessfully completed all operations!")
        else:
            print("\nOperation was aborted or failed.")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving performance data...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Saving performance data...")
    finally:
        # Always save performance data
        if attempt_data['attempts']:
            model_name = get_model_name(args.model)
            save_performance_data(model_name)

if __name__ == "__main__":
    main()
