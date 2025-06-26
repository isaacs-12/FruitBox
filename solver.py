from ocr_grid import print_grid
import numpy as np
from typing import List, Tuple
import time
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import os

def find_rectangles_zero_inclusive_greedy(grid):
    rows, cols = grid.shape
    current_grid = np.copy(grid)
    found_rectangles = []

    while True:
        possible_rects_in_pass = []
        
        # 1. Generate all currently valid rectangles based on the new definition
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        
                        current_sum_of_non_zeros = 0
                        has_at_least_one_non_zero = False # Flag to ensure it's not an all-zero rect
                        
                        # Iterate through cells within the potential rectangle
                        for r_cell in range(r1, r2 + 1):
                            for c_cell in range(c1, c2 + 1):
                                if current_grid[r_cell, c_cell] != 0:
                                    current_sum_of_non_zeros += current_grid[r_cell, c_cell]
                                    has_at_least_one_non_zero = True
                        
                        # Check if the sum of non-zero digits is 10 and it's not an empty rect
                        if has_at_least_one_non_zero and current_sum_of_non_zeros == 10:
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            possible_rects_in_pass.append((area, (r1, c1, r2, c2)))
        
        if not possible_rects_in_pass:
            # No more rectangles can be found in the current grid state
            break

        # 2. Sort potential rectangles by area (descending)
        possible_rects_in_pass.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Select the largest available rectangle and apply changes
        selected_one_this_pass = False
        for area, (r1, c1, r2, c2) in possible_rects_in_pass:
            # Re-validate: ensure it's still valid *in the current grid state*
            # (another rectangle might have been selected in this pass)
            recheck_sum_of_non_zeros = 0
            recheck_has_at_least_one_non_zero = False
            for r_cell in range(r1, r2 + 1):
                for c_cell in range(c1, c2 + 1):
                    if current_grid[r_cell, c_cell] != 0:
                        recheck_sum_of_non_zeros += current_grid[r_cell, c_cell]
                        recheck_has_at_least_one_non_zero = True
            
            if recheck_has_at_least_one_non_zero and recheck_sum_of_non_zeros == 10:
                found_rectangles.append((r1, c1, r2, c2))
                
                # Zero out the selected rectangle in the current_grid
                for r_cell in range(r1, r2 + 1):
                    for c_cell in range(c1, c2 + 1):
                        current_grid[r_cell, c_cell] = 0
                
                selected_one_this_pass = True
                break # Restart outer loop (new pass)
        
        if not selected_one_this_pass:
            # This can happen if all rectangles generated in possible_rects_in_pass
            # become invalid due to earlier selections within the same pass (if it's not restarted after each selection).
            # But with the `break` after selection, it should imply no new valid ones could be found.
            break

    return found_rectangles, current_grid.copy()

def find_rectangles_digit_priority_greedy(grid, max_retries: int = 3) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, int]:
    """
    Find rectangles that sum to 10, prioritizing those with more digits.
    Includes retry logic to find optimal solutions.
    
    Args:
        grid: 2D numpy array containing the digit grid
        max_retries: Maximum number of attempts to find a better solution
        
    Returns:
        tuple: (list of rectangles, final grid state, number of digits cleared)
    """
    def count_non_zero_digits(r1: int, c1: int, r2: int, c2: int, current_grid: np.ndarray) -> int:
        """Count the number of non-zero digits in a rectangle."""
        count = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if current_grid[r, c] != 0:
                    count += 1
        return count
    
    def find_best_rectangles(current_grid: np.ndarray, attempt: int) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, int]:
        """Find the best set of rectangles for the current grid state."""
        rows, cols = current_grid.shape
        found_rectangles = []
        grid_copy = np.copy(current_grid)
        total_digits_cleared = 0
        is_first_choice = True  # Track if this is the first rectangle choice
        
        while True:
            possible_rects = []
            
            # Generate all valid rectangles
            for r1 in range(rows):
                for c1 in range(cols):
                    for r2 in range(r1, rows):
                        for c2 in range(c1, cols):
                            # Calculate sum and count of non-zero digits
                            current_sum = 0
                            digit_count = 0
                            for r in range(r1, r2 + 1):
                                for c in range(c1, c2 + 1):
                                    if grid_copy[r, c] != 0:
                                        current_sum += grid_copy[r, c]
                                        digit_count += 1
                            
                            # Check if valid rectangle (sums to 10 and has at least one digit)
                            if digit_count > 0 and current_sum == 10:
                                # For first choice in first attempt, heavily randomize
                                if is_first_choice and attempt == 0:
                                    # Use a much larger random factor for first choice
                                    base_score = digit_count
                                    random_factor = np.random.RandomState(attempt).uniform(0, 2.0)
                                    score = base_score + random_factor
                                else:
                                    # Normal scoring for other choices
                                    base_score = digit_count
                                    random_factor = np.random.RandomState(attempt).uniform(0, 0.1)
                                    score = base_score + random_factor
                                possible_rects.append((score, (r1, c1, r2, c2)))
            
            if not possible_rects:
                break
            
            # Sort by score in descending order
            possible_rects.sort(key=lambda x: x[0], reverse=True)
            
            # For first choice in first attempt, consider many more candidates
            if is_first_choice and attempt == 0:
                num_candidates = min(10, len(possible_rects))  # Consider up to 10 candidates
            else:
                # For other choices, use normal candidate selection
                num_candidates = min(3, 1 + attempt)
            
            candidates = possible_rects[:num_candidates]
            
            # Randomly select from candidates
            selected_idx = np.random.RandomState(attempt + (0 if is_first_choice else 1000)).randint(0, len(candidates))
            _, (r1, c1, r2, c2) = candidates[selected_idx]
            
            # Verify it's still valid
            current_sum = 0
            digit_count = 0
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if grid_copy[r, c] != 0:
                        current_sum += grid_copy[r, c]
                        digit_count += 1
            
            if digit_count > 0 and current_sum == 10:
                found_rectangles.append((r1, c1, r2, c2))
                total_digits_cleared += digit_count
                
                # Zero out the selected rectangle
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        grid_copy[r, c] = 0
                
                is_first_choice = False  # No longer the first choice
            else:
                # If the selected rectangle is no longer valid, try the next one
                continue
        
        return found_rectangles, grid_copy, total_digits_cleared
    
    # Try multiple times to find the best solution
    best_solution = None
    best_digits_cleared = 0
    best_final_grid = None
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    for attempt in range(max_retries):
        # For each attempt, start with a fresh copy of the grid
        current_grid = np.copy(grid)
        
        # Find rectangles for this attempt
        rectangles, final_grid, digits_cleared = find_best_rectangles(current_grid, attempt)
        
        # Update best solution if this attempt cleared more digits
        if digits_cleared > best_digits_cleared:
            best_solution = rectangles
            best_digits_cleared = digits_cleared
            best_final_grid = final_grid
            print(f"Attempt {attempt + 1}: Found better solution with {digits_cleared} digits cleared")
    
    return best_solution, best_final_grid, best_digits_cleared

def solve_rectangles_backtracking(initial_grid, time_limit=10.0):
    rows, cols = initial_grid.shape
    best_solution = []
    final_grid = None
    best_count = 0
    start_time = time.time()
    is_timeout = False

    def get_non_zero_sum(grid_slice):
        return np.sum(grid_slice[grid_slice != 0])

    def find_best_recursive(current_grid, current_rectangles, digits_cleared):
        nonlocal best_solution, best_count, final_grid, is_timeout

        # Time limit check - if we hit timeout, just return current best
        if time.time() - start_time > time_limit:
            print("Search timed out, returning best solution found so far")
            is_timeout = True
            return

        # Prune if this path can't possibly beat the best
        remaining = np.count_nonzero(current_grid)
        if digits_cleared + remaining <= best_count:
            return

        # Find all valid rectangles, prioritize by number of non-zero digits
        possible_rects = []
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        rect_slice = current_grid[r1:r2+1, c1:c2+1]
                        if np.all(rect_slice == 0):
                            continue
                        current_sum = get_non_zero_sum(rect_slice)
                        if current_sum == 10:
                            nonzero_count = np.count_nonzero(rect_slice)
                            possible_rects.append((nonzero_count, (r1, c1, r2, c2)))

        if not possible_rects:
            if digits_cleared > best_count:
                print(f"New best solution found with {digits_cleared} digits cleared")
                best_solution = list(current_rectangles)
                best_count = digits_cleared
                final_grid = np.copy(current_grid)
            return

        # Sort by most non-zero digits cleared (descending)
        possible_rects.sort(key=lambda x: x[0], reverse=True)

        for nonzero_count, (r1, c1, r2, c2) in possible_rects:
            # Check time limit before each new branch
            if time.time() - start_time > time_limit:
                is_timeout = True
                return
                
            next_grid = np.copy(current_grid)
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    next_grid[r, c] = 0
            current_rectangles.append((r1, c1, r2, c2))
            find_best_recursive(next_grid, current_rectangles, digits_cleared + nonzero_count)
            current_rectangles.pop()

    find_best_recursive(initial_grid, [], 0)
    return best_solution, final_grid, best_count, is_timeout

def solve_rectangles_model(initial_grid: np.ndarray, model_path: str) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, int]:
    """
    Solve a grid using a trained model.
    
    Args:
        initial_grid: 2D numpy array containing the digit grid
        model_path: Path to the trained model file
        
    Returns:
        tuple: (list of rectangles, final grid state, number of digits cleared)
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    # Create a simple environment for solving
    class SimpleRectangleEnv(gym.Env):
        def __init__(self, initial_grid):
            super().__init__()
            self.grid_shape = initial_grid.shape
            self.rows, self.cols = self.grid_shape
            self.observation_space = spaces.Box(low=0, high=9, shape=(self.rows, self.cols, 1), dtype=np.uint8)
            self.max_possible_actions = ((self.rows * (self.rows + 1) // 2) * (self.cols * (self.cols + 1) // 2))
            self.action_space = spaces.Discrete(self.max_possible_actions)
            self.initial_grid = initial_grid
            self.grid = None
            self.rectangle_id_map = {}
            self._populate_rectangle_id_map()
            
        def _populate_rectangle_id_map(self):
            """Create a mapping from action IDs to rectangle coordinates."""
            action_id = 0
            for r1 in range(self.rows):
                for c1 in range(self.cols):
                    for r2 in range(r1, self.rows):
                        for c2 in range(c1, self.cols):
                            self.rectangle_id_map[action_id] = (r1, c1, r2, c2)
                            action_id += 1
        
        def _get_rectangle_from_action(self, action):
            """Get rectangle coordinates from action ID."""
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
            self.grid = np.copy(self.initial_grid)
            return self.grid.reshape(self.rows, self.cols, 1), {"action_mask": self._get_valid_actions()}
        
        def step(self, action):
            r1, c1, r2, c2 = self._get_rectangle_from_action(action)
            
            # Print the rectangle being selected
            print(f"Selecting rectangle: ({r1},{c1}) to ({r2},{c2})")
            
            # Calculate reward (area of rectangle)
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            
            # Zero out the rectangle
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    self.grid[r, c] = 0
            
            # Check if done (no more valid actions)
            valid_actions = self._get_valid_actions()
            done = not np.any(valid_actions)
            
            if done:
                print("No more valid actions available")
            
            return self.grid.reshape(self.rows, self.cols, 1), area, done, False, {"action_mask": valid_actions}
    
    try:
        # Load the model
        print("Loading PPO model...")
        model = PPO.load(model_path)
        print("Model loaded successfully")
        
        # Create environment and solve
        print("Creating environment...")
        env = SimpleRectangleEnv(initial_grid)
        obs, info = env.reset()
        print("Environment created and reset")
        
        done = False
        total_reward = 0
        rectangles = []
        step_count = 0
        max_steps = 100  # Safety limit to prevent infinite loops
        
        while not done and step_count < max_steps:
            step_count += 1
            print(f"\nStep {step_count}")
            
            action_mask = info["action_mask"]
            valid_action_count = np.sum(action_mask)
            print(f"Valid actions available: {valid_action_count}")
            
            if not np.any(action_mask):
                print("No valid actions available, breaking")
                break
            
            # Get action from model and ensure it's valid
            print("Getting action from model...")
            action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            
            # If the model's action is invalid, find the first valid action
            if not action_mask[action]:
                print(f"Model selected invalid action {action}, finding first valid action")
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
                else:
                    print("No valid actions found")
                    break
            
            print(f"Selected action: {action}")
            
            # Take step
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
            # Store the rectangle
            r1, c1, r2, c2 = env._get_rectangle_from_action(action)
            rectangles.append((r1, c1, r2, c2))
            print(f"Total rectangles found: {len(rectangles)}")
        
        if step_count >= max_steps:
            print(f"Warning: Reached maximum step limit of {max_steps}")
        
        # Calculate number of digits cleared
        num_digits_cleared = np.count_nonzero(initial_grid) - np.count_nonzero(env.grid)
        print(f"\nFinal results:")
        print(f"Total steps taken: {step_count}")
        print(f"Total rectangles found: {len(rectangles)}")
        print(f"Total digits cleared: {num_digits_cleared}")
        
        return rectangles, env.grid, num_digits_cleared
        
    except Exception as e:
        print(f"Error during model solving: {str(e)}")
        raise

# Example Usage:
initial_grid = np.array([
    [8, 8, 9, 1, 1, 8, 2, 8, 6, 9, 5, 2, 1, 3, 6, 5, 7],
    [5, 8, 4, 9, 3, 4, 5, 4, 7, 9, 5, 8, 8, 9, 9, 2, 9],
    [7, 3, 9, 4, 5, 3, 3, 1, 6, 9, 4, 4, 2, 5, 6, 6, 2],
    [5, 1, 9, 2, 2, 8, 4, 2, 5, 4, 8, 9, 3, 4, 8, 1, 8],
    [8, 6, 8, 2, 6, 7, 3, 4, 7, 4, 1, 2, 3, 3, 1, 2, 4],
    [8, 2, 7, 3, 4, 3, 4, 2, 1, 6, 5, 9, 1, 3, 7, 6, 6],
    [6, 7, 6, 5, 6, 8, 9, 9, 9, 8, 2, 5, 1, 5, 2, 7, 9],
    [1, 7, 8, 4, 5, 9, 6, 2, 5, 7, 4, 1, 5, 8, 5, 1, 7],
    [2, 1, 6, 6, 7, 9, 9, 6, 4, 5, 8, 2, 7, 9, 1, 7, 2],
    [2, 9, 5, 7, 1, 6, 1, 8, 4, 2, 3, 6, 6, 6, 7, 3, 7]
])

def __main__():
    rectangles, final_grid, num_digits_cleared = solve_rectangles_model(initial_grid, "models/model_1_500k.zip")
    print("Rectangles:", rectangles)
    print("Final grid:")
    print_grid(final_grid)
    print("Number of digits cleared:", num_digits_cleared)

if __name__ == "__main__":
    __main__()

# print("Initial grid:")
# print_grid(initial_grid)
# found_rects_zero_inclusive, final_grid_greedy = find_rectangles_zero_inclusive_greedy(initial_grid)
# print("Found rectangles (zero-inclusive greedy):", found_rects_zero_inclusive)
# print("Number of rectangles found (zero-inclusive greedy):", len(found_rects_zero_inclusive))
# print("Final grid (greedy):")
# print_grid(final_grid_greedy)
# 
# print("Number of apples found (greedy):", np.count_nonzero(initial_grid)-np.count_nonzero(final_grid_greedy))
