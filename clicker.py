import pyautogui
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import sys
import cv2
import os
from ocr_grid import process_image_to_grid
from datetime import datetime

from utils import print_verbose

# Safety feature: move mouse to corner to stop
pyautogui.FAILSAFE = True
TIME_LIMIT = 30
DRAG_DURATION = 0.1
DRAG_PAUSE = 0.2

class ScreenOffset:
    def __init__(self, x: int = 70, y: int = 175):
        self.x = x
        self.y = y
    
    def adjust_position(self, x: int, y: int) -> Tuple[int, int]:
        """Adjust coordinates by adding the offset."""
        return (x + self.x, y + self.y)

def get_grid_corners(results: List[Dict]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Calculate the top-left and bottom-right corners of the grid based on digit positions.
    
    Args:
        results: List of digit results from process_image_to_grid
        
    Returns:
        tuple: ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
    """
    if not results:
        raise ValueError("No results provided")
    
    # Get all x and y coordinates
    x_coords = [r['position'][0] for r in results]  # x coordinates
    y_coords = [r['position'][1] for r in results]  # y coordinates
    widths = [r['position'][2] for r in results]    # widths
    heights = [r['position'][3] for r in results]   # heights
    
    # Calculate grid corners
    top_left = (min(x_coords), min(y_coords))
    bottom_right = (max(x + w for x, w in zip(x_coords, widths)),
                   max(y + h for y, h in zip(y_coords, heights)))
    
    return top_left, bottom_right

def get_cell_bounds(grid_corners: Tuple[Tuple[int, int], Tuple[int, int]], 
                   row: int, col: int) -> Tuple[int, int, int, int]:
    """
    Calculate the bounds of a specific cell in the grid.
    
    Args:
        grid_corners: ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
        row: Row index (0-9)
        col: Column index (0-16)
        
    Returns:
        tuple: (x, y, width, height) of the cell
    """
    top_left, bottom_right = grid_corners
    
    # Calculate cell dimensions
    total_width = bottom_right[0] - top_left[0]
    total_height = bottom_right[1] - top_left[1]
    cell_width = total_width / 17  # 17 columns
    cell_height = total_height / 10  # 10 rows
    
    # Calculate cell bounds
    x = top_left[0] + (col * cell_width)
    y = top_left[1] + (row * cell_height)
    
    return (int(x), int(y), int(cell_width), int(cell_height))

def draw_rectangle(x: int, y: int, width: int, height: int, 
                  screen_offset: ScreenOffset, duration: float = 0.5) -> None:
    """
    Draw a rectangle by clicking and dragging from top-left to bottom-right corner.
    Includes precise positioning and pauses for better click registration.
    
    Args:
        x, y: Top-left corner coordinates
        width, height: Rectangle dimensions
        screen_offset: ScreenOffset object for coordinate adjustment
        duration: Time to spend drawing the rectangle
    """
    # Adjust coordinates for screen offset
    start_x, start_y = screen_offset.adjust_position(x, y)
    end_x, end_y = screen_offset.adjust_position(x + width, y + height)
    
    # Move to start position quickly but precisely
    pyautogui.moveTo(start_x, start_y, duration=DRAG_DURATION)
    
    # Pause briefly to ensure stable position
    time.sleep(DRAG_PAUSE)
    
    # Click and hold
    pyautogui.mouseDown()
    
    # Pause briefly after clicking
    time.sleep(DRAG_PAUSE)
    
    # Drag to end position quickly but precisely
    pyautogui.moveTo(end_x, end_y, duration=DRAG_DURATION)
    
    # Pause briefly before releasing
    time.sleep(DRAG_PAUSE)
    
    # Release mouse button
    pyautogui.mouseUp()
    
    # Small pause to ensure the click is registered
    time.sleep(0.1)

def create_overlay(image_path: str, grid_corners: Tuple[Tuple[int, int], Tuple[int, int]], 
                  solution: List[Union[Tuple[int, int], Tuple[int, int, int, int]]], screen_offset: ScreenOffset,
                  padding: int = 5) -> np.ndarray:
    """
    Create a visual overlay of the rectangles on the input image.
    Note: Screen offset is only used for display purposes, not applied to coordinates.
    
    Args:
        image_path: Path to the input image
        grid_corners: Grid corners from get_grid_corners
        solution: List of either:
            - (row, col) tuples for individual cells
            - (r1, c1, r2, c2) tuples for rectangles
        screen_offset: ScreenOffset object (only used for display info)
        padding: Padding around rectangles
        
    Returns:
        numpy.ndarray: Image with rectangles drawn on it
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create a copy for drawing
    overlay = image.copy()
    
    # Draw each rectangle or cell
    for item in solution:
        if len(item) == 2:  # Individual cell (row, col)
            row, col = item
            # Get cell bounds
            x, y, width, height = get_cell_bounds(grid_corners, row, col)
        else:  # Rectangle (r1, c1, r2, c2)
            r1, c1, r2, c2 = item
            # Get bounds for the rectangle
            x1, y1, w1, h1 = get_cell_bounds(grid_corners, r1, c1)
            x2, y2, w2, h2 = get_cell_bounds(grid_corners, r2, c2)
            x = min(x1, x2)
            y = min(y1, y2)
            width = max(x1 + w1, x2 + w2) - x
            height = max(y1 + h1, y2 + h2) - y
        
        # Add padding
        x -= padding
        y -= padding
        width += 2 * padding
        height += 2 * padding
        
        # Draw rectangle (without screen offset)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Add coordinates
        if len(item) == 2:
            cv2.putText(overlay, f"({row},{col})", (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(overlay, f"({r1},{c1})-({r2},{c2})", (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add grid corners and offset info (for reference only)
    top_left, bottom_right = grid_corners
    cv2.putText(overlay, f"Grid: {top_left} -> {bottom_right}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(overlay, f"Screen offset: ({screen_offset.x}, {screen_offset.y})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(overlay, "Note: Screen offset will be applied during clicking", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return overlay

def take_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    Take a screenshot and save it to a temporary file.
    
    Args:
        region: Optional (left, top, width, height) tuple to capture a specific region
        
    Returns:
        str: Path to the saved screenshot
    """
    # Create screenshots directory if it doesn't exist
    os.makedirs('screenshots', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'screenshots/screenshot_{timestamp}.png'
    
    # Take screenshot
    if region:
        screenshot = pyautogui.screenshot(region=region)
    else:
        screenshot = pyautogui.screenshot()
    
    # Save screenshot
    screenshot.save(filename)
    print_verbose(f"Screenshot saved to {filename}")
    return filename

def process_and_click(image_path: Optional[str] = None, 
                     solution: Union[List[Union[Tuple[int, int], Tuple[int, int, int, int]]], str] = None,
                     screen_offset: Optional[Union[ScreenOffset, Tuple[int, int]]] = None,
                     method: str = 'template',
                     duration: float = TIME_LIMIT,
                     preview: bool = True,
                     preview_path: str = "preview_overlay.png",
                     screenshot_region: Optional[Tuple[int, int, int, int]] = None,
                     results: Optional[List[Dict]] = None,
                     verbose: bool = False) -> bool:
    """
    Process an image and click rectangles around the solution cells.
    This is the main function to be called from other Python files.
    
    Args:
        image_path: Path to the input image, or None to take a screenshot
        solution: Either:
            - A list of (row, col) tuples for individual cells
            - A list of (r1, c1, r2, c2) tuples for rectangles
            - Path to a solution file
        screen_offset: Either a ScreenOffset object, (x,y) tuple, or None
        method: Method to use for digit recognition ('template', 'ocr', or 'pixel')
        duration: Total time to spend clicking (in seconds)
        preview: Whether to show a preview of the rectangles before clicking
        preview_path: Where to save the preview image
        screenshot_region: Optional (left, top, width, height) tuple to capture a specific region
        results: Optional pre-computed results from process_image_to_grid
        verbose: Whether to print verbose output
    Returns:
        bool: True if clicking was completed, False if aborted
    """
    # Take screenshot if no image path provided
    if image_path is None:
        print_verbose("Taking screenshot...")
        image_path = take_screenshot(screenshot_region)
    
    # Convert screen_offset to ScreenOffset object if needed
    if isinstance(screen_offset, tuple):
        screen_offset = ScreenOffset(*screen_offset)
    elif screen_offset is None:
        screen_offset = ScreenOffset()
    
    # Load solution from file if string is provided
    if isinstance(solution, str):
        solution = load_solution_from_file(solution)
    
    try:
        # Process the image to get grid
        print_verbose("Processing image...")
        if results is None:
            _, results = process_image_to_grid(image_path, method)
        
        # Get grid corners
        grid_corners = get_grid_corners(results)
        print_verbose(f"Grid corners: {grid_corners}")
        print_verbose(f"Screen offset: ({screen_offset.x}, {screen_offset.y})")
        
        # Create and show overlay if preview is enabled
        if preview:
            overlay = create_overlay(image_path, grid_corners, solution, screen_offset)
            cv2.imwrite(preview_path, overlay)
            print_verbose(f"\nPreview saved to {preview_path}")
            print_verbose("Please verify the rectangle positions before proceeding.")
        
        # Calculate time per rectangle (excluding pauses)
        num_rectangles = len(solution)
        time_per_rectangle = (duration - (num_rectangles * 0.8)) / num_rectangles if num_rectangles > 0 else 0
        
        # Give user time to switch to the correct window
        print_verbose(f"Starting in 0.5 seconds... Move mouse to corner to stop.")
        time.sleep(0.5)
        
        # Click once at the top-left corner of the grid to ensure window focus
        top_left, _ = grid_corners
        focus_x, focus_y = screen_offset.adjust_position(*top_left)
        print_verbose("Clicking to focus window...")
        pyautogui.moveTo(focus_x, focus_y, duration=DRAG_DURATION)
        time.sleep(DRAG_PAUSE)
        pyautogui.click()
        time.sleep(DRAG_PAUSE)  # Wait for window to respond
        
        start_time = time.time()
        # Process solution in order
        for i, item in enumerate(solution):
            # Check if we've exceeded the duration
            if time.time() - start_time >= duration:
                print_verbose("Time limit reached")
                break
                
            if len(item) == 2:  # Individual cell (row, col)
                row, col = item
                # Get cell bounds
                x, y, width, height = get_cell_bounds(grid_corners, row, col)
                print_verbose(f"Drawing rectangle {i+1}/{num_rectangles} at cell ({row}, {col})")
            else:  # Rectangle (r1, c1, r2, c2)
                r1, c1, r2, c2 = item
                # Get bounds for the rectangle
                x1, y1, w1, h1 = get_cell_bounds(grid_corners, r1, c1)
                x2, y2, w2, h2 = get_cell_bounds(grid_corners, r2, c2)
                x = min(x1, x2)
                y = min(y1, y2)
                width = max(x1 + w1, x2 + w2) - x
                height = max(y1 + h1, y2 + h2) - y
                print_verbose(f"Drawing rectangle {i+1}/{num_rectangles} from ({r1}, {c1}) to ({r2}, {c2})")
            
            # Draw rectangle with screen offset
            draw_rectangle(x, y, width, height, screen_offset, duration=time_per_rectangle)
            
            # Small pause between rectangles (already included in draw_rectangle)
            # time.sleep(time_per_rectangle * 0.2)  # Removed since pauses are now in draw_rectangle
        
        return True
            
    except Exception as e:
        print_verbose(f"Error: {str(e)}")
        raise

def load_solution_from_file(solution_file: str) -> List[Tuple[int, int]]:
    """Load solution from a file."""
    solution = []
    with open(solution_file, 'r') as f:
        for line in f:
            # Expect format: "row col" or "row,col"
            if ',' in line:
                row, col = map(int, line.strip().split(','))
            else:
                row, col = map(int, line.strip().split())
            solution.append((row, col))
    return solution

def main():
    """Command-line interface for the script."""
    if len(sys.argv) < 2:
        print_verbose("Usage: python clicker.py [image_path] <solution_file> [--offset x,y] [--method template|ocr|pixel] [--no-preview] [--region left,top,width,height]")
        print_verbose("If image_path is omitted, a screenshot will be taken.")
        sys.exit(1)
    
    # Parse arguments
    image_path = None
    solution_file = None
    method = 'template'  # default method
    screen_offset = ScreenOffset()  # default offset
    preview = True  # default to showing preview
    screenshot_region = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            # Handle flags
            if sys.argv[i] == '--offset':
                if i + 1 >= len(sys.argv):
                    print_verbose("Error: --offset requires x,y coordinates")
                    sys.exit(1)
                try:
                    x, y = map(int, sys.argv[i + 1].split(','))
                    screen_offset = ScreenOffset(x, y)
                    i += 2
                except ValueError:
                    print_verbose("Error: --offset coordinates must be integers in format x,y")
                    sys.exit(1)
            elif sys.argv[i] == '--method':
                if i + 1 >= len(sys.argv):
                    print_verbose("Error: --method requires a value")
                    sys.exit(1)
                method = sys.argv[i + 1]
                if method not in ['template', 'ocr', 'pixel']:
                    print_verbose("Error: method must be one of: template, ocr, pixel")
                    sys.exit(1)
                i += 2
            elif sys.argv[i] == '--no-preview':
                preview = False
                i += 1
            elif sys.argv[i] == '--region':
                if i + 1 >= len(sys.argv):
                    print_verbose("Error: --region requires left,top,width,height coordinates")
                    sys.exit(1)
                try:
                    left, top, width, height = map(int, sys.argv[i + 1].split(','))
                    screenshot_region = (left, top, width, height)
                    i += 2
                except ValueError:
                    print_verbose("Error: --region coordinates must be integers in format left,top,width,height")
                    sys.exit(1)
            else:
                print_verbose(f"Error: unknown argument {sys.argv[i]}")
                sys.exit(1)
        else:
            # Handle positional arguments
            if image_path is None and not sys.argv[i].startswith('--'):
                # First non-flag argument is image path (optional)
                image_path = sys.argv[i]
                i += 1
            elif solution_file is None and not sys.argv[i].startswith('--'):
                # Second non-flag argument is solution file (required)
                solution_file = sys.argv[i]
                i += 1
            else:
                i += 1
    
    if solution_file is None:
        print_verbose("Error: solution file is required")
        sys.exit(1)
    
    try:
        process_and_click(image_path, solution_file, screen_offset, method, 
                         preview=preview, screenshot_region=screenshot_region)
    except Exception as e:
        print_verbose(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
