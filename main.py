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

test_image = "test_data/test_4.png"

RESET_LOC = (78,1110)
PLAY_LOC = (370,644)

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

def process_and_solve(region: Optional[Tuple[int, int, int, int]] = None,
                     screen_offset: Optional[Union[ScreenOffset, Tuple[int, int]]] = None,
                     method: str = 'pixel',
                     duration: float = 30.0,
                     preview: bool = True,
                     verbose: bool = False,
                     use_test_image: bool = False) -> bool:
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
    Returns:
        bool: True if clicking was completed, False if aborted
    """


    timer = time.time()
    
    # Process the image and get grid
    print_verbose("Processing image...", verbose)
    image_path = test_image if use_test_image else None
    grid, results = process_image_to_grid(image_path, method, screenshot_region=region, verbose=verbose)
    print_verbose("Extracted grid:", verbose)
    print_grid(grid)
    
    # Save the grid for later training
    save_grid(grid)
    
    # Find solution
    print_verbose("Finding solution...", verbose)
    # solutions, final_grid, num_digits_cleared = solve_rectangles_backtracking(grid)
    solutions, final_grid, num_digits_cleared = solve_rectangles_model(grid, "models/model_1_500k.zip")
    print_verbose("Found rectangles:", verbose)
    for r1, c1, r2, c2 in solutions:
        print_verbose(f"({r1},{c1}) to ({r2},{c2})", verbose)
    
    elapsed_time = round(time.time() - timer, 3)
    print_verbose(f"\nTime taken to process and solve: {elapsed_time} seconds", verbose)
    print(f"Number of apples found: {num_digits_cleared}", verbose) # Always printed

    if num_digits_cleared < 130:
        print(f"Not enough apples, only found {num_digits_cleared} apples. Resting...", verbose)
        reset_game()
        main()
    
    # Click the solution
    print_verbose("\nClicking solution...", verbose)
    success = process_and_click(
        image_path=image_path,  # Will take a new screenshot
        solution=solutions,
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
    region = (10, 172, 1431, 842)  # (left, top, width, height)
    screen_offset = (10, 172)     # (x, y) offset

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Use test image instead of screenshot')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    # Process and solve
    success = process_and_solve(
        region=region,
        screen_offset=screen_offset,
        method='pixel',
        duration=120.0,
        preview=True,
        verbose=args.verbose,
        use_test_image=args.test
    )
    
    if success:
        print("\nSuccessfully completed all operations!")
    else:
        print("\nOperation was aborted or failed.")

if __name__ == "__main__":
    main()
