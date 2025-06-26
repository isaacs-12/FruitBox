import cv2
import numpy as np
import os
from PIL import Image
import sys
import pytesseract  # Add pytesseract import
from typing import Optional, Tuple, List, Dict, Union

from utils import print_verbose

class Digit:
    def __init__(self, image, contour, inner_contour, bbox):
        self.image = image
        self.contour = contour
        self.inner_contour = inner_contour
        self.bbox = bbox

    def __str__(self) -> str:
        return f"Digit(image={self.image}, contour={self.contour}, inner_contour={self.inner_contour}, bbox={self.bbox})"

def crop_and_normalize_digit(digit_img, target_size=(32, 32)):
    """
    Crop the digit image to its actual bounds and normalize to target size.
    Returns the cropped and normalized image, and the original bounds.
    """
    # Find the bounds of the white pixels (the actual digit)
    white_pixels = np.where(digit_img == 255)
    if len(white_pixels[0]) == 0:  # No white pixels found
        return digit_img, (0, 0, digit_img.shape[1], digit_img.shape[0])
        
    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    
    # Add a small padding around the digit
    padding = 2
    min_y = max(0, min_y - padding)
    min_x = max(0, min_x - padding)
    max_y = min(digit_img.shape[0], max_y + padding)
    max_x = min(digit_img.shape[1], max_x + padding)
    
    # Crop to the digit bounds
    cropped = digit_img[min_y:max_y, min_x:max_x]
    
    # Normalize to target size
    normalized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return normalized, (min_x, min_y, max_x - min_x, max_y - min_y)

def find_matching_template(digit_img, templates, verbose: bool = False):
    """
    Compares a digit image against template images to find the best matching digit.
    Uses TM_SQDIFF_NORMED where 0 is perfect match and 1 is worst match.
    """
    if digit_img is None:
        print_verbose("Error: No image provided for matching", verbose)
        return 0, 0.0

    # For SQDIFF, lower is better, so we initialize to worst possible score
    best_score = float('inf')  # Changed from -inf to inf
    best_match = 0
    second_best_score = float('inf')  # Changed from -inf to inf
    second_best_match = 0

    # Calculate some basic features of the digit
    h, w = digit_img.shape
    aspect_ratio = w / h if h > 0 else 0
    white_pixels = np.sum(digit_img == 255)
    total_pixels = h * w
    fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    print_verbose(f"Digit features - Aspect ratio: {aspect_ratio:.2f}, Fill ratio: {fill_ratio:.2f}", verbose)

    # Iterate through each template
    for digit, template in templates.items():
        if template.image is None:
            print_verbose(f"Warning: No image found for template {digit}. Skipping.", verbose)
            continue

        try:
            # Templates are already normalized to the same size as digits
            template_img = template.image

            # Calculate template features
            t_h, t_w = template_img.shape
            t_aspect_ratio = t_w / t_h if t_h > 0 else 0
            t_white_pixels = np.sum(template_img == 255)
            t_total_pixels = t_h * t_w
            t_fill_ratio = t_white_pixels / t_total_pixels if t_total_pixels > 0 else 0

            # Use template matching on raw images
            result = cv2.matchTemplate(digit_img, template_img, cv2.TM_SQDIFF_NORMED)
            score = np.min(result)  # Changed from max to min since lower is better

            # Create a visualization of the image matching
            # comparison = np.hstack((digit_img, template_img))
            # save_debug_step(comparison, f"template_match_{digit}_score_{score:.2f}")

            # Update best and second best matches (lower is better for SQDIFF)
            if score < best_score:  # Changed from > to < since lower is better
                second_best_score = best_score
                second_best_match = best_match
                best_score = score
                best_match = digit
            elif score < second_best_score:  # Changed from > to < since lower is better
                second_best_score = score
                second_best_match = digit

        except Exception as e:
            print_verbose(f"Error matching template {digit}: {e}. Skipping.", verbose)
            continue

    # Normal threshold for all digits
    if best_score > 0.5:  # Changed from 0.5 to 0.5 but inverted meaning
        print_verbose(f"No confident match found (best score: {best_score:.2f})", verbose)
        return 0, 1.0 - best_score  # Convert to confidence score

    print_verbose(f"Best match: digit {best_match} with score {best_score:.2f} (confidence: {1.0 - best_score:.2f})", verbose)
    print_verbose(f"Second best: digit {second_best_match} with score {second_best_score:.2f} (confidence: {1.0 - second_best_score:.2f})", verbose)
    return best_match, 1.0 - best_score  # Convert to confidence score

def find_most_similar_template(image_num, digit_img, templates, verbose: bool = False):
    """
    Find the most similar template by comparing pixel values directly.
    Uses normalized pixel-wise difference and structural similarity index (SSIM).
    
    Args:
        digit_img: The digit image to compare (white digit on black background)
        templates: Dictionary of template Digit objects
        
    Returns:
        tuple: (best_matching_digit, confidence)
    """
    if digit_img is None:
        print_verbose("Error: No image provided for comparison", verbose)
        return 0, 0.0

    best_score = float('-inf')  # Higher is better for similarity
    best_match = 0
    second_best_score = float('-inf')
    second_best_match = 0

    # Calculate some basic features of the digit
    h, w = digit_img.shape
    aspect_ratio = w / h if h > 0 else 0
    white_pixels = np.sum(digit_img == 255)
    total_pixels = h * w
    fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    print_verbose(f"Digit features - Aspect ratio: {aspect_ratio:.2f}, Fill ratio: {fill_ratio:.2f}", verbose)

    # Resize digit image to match template size (32x32)
    digit_img_resized = cv2.resize(digit_img, (32, 32), interpolation=cv2.INTER_AREA)

    # Iterate through each template
    for digit, template in templates.items():
        if template.image is None:
            print_verbose(f"Warning: No image found for template {digit}. Skipping.", verbose)
            continue

        try:
            template_img = template.image
            
            # Calculate template features
            t_h, t_w = template_img.shape
            t_aspect_ratio = t_w / t_h if t_h > 0 else 0
            t_white_pixels = np.sum(template_img == 255)
            t_total_pixels = t_h * t_w
            t_fill_ratio = t_white_pixels / t_total_pixels if t_total_pixels > 0 else 0

            # 1. Calculate pixel-wise difference (normalized)
            diff = np.abs(digit_img_resized.astype(float) - template_img.astype(float))
            pixel_diff_score = 1.0 - (np.sum(diff) / (255.0 * t_total_pixels))  # Normalize to 0-1
            
            # 2. Calculate structural similarity (SSIM)
            # Convert to float and normalize to 0-1 range
            digit_norm = digit_img_resized.astype(float) / 255.0
            template_norm = template_img.astype(float) / 255.0
            
            # Calculate mean and variance
            digit_mean = np.mean(digit_norm)
            template_mean = np.mean(template_norm)
            digit_var = np.var(digit_norm)
            template_var = np.var(template_norm)
            
            # Calculate covariance
            covar = np.mean((digit_norm - digit_mean) * (template_norm - template_mean))
            
            # SSIM constants
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            # Calculate SSIM
            numerator = (2 * digit_mean * template_mean + C1) * (2 * covar + C2)
            denominator = (digit_mean ** 2 + template_mean ** 2 + C1) * (digit_var + template_var + C2)
            ssim_score = numerator / denominator if denominator != 0 else 0
            
            # Combine scores (weighted average)
            similarity_score = 0.7 * pixel_diff_score + 0.3 * ssim_score
            
            # Update best and second best matches
            if similarity_score > best_score:
                second_best_score = best_score
                second_best_match = best_match
                best_score = similarity_score
                best_match = digit
            elif similarity_score > second_best_score:
                second_best_score = similarity_score
                second_best_match = digit

        except Exception as e:
            print_verbose(f"Error comparing template {digit}: {str(e)}. Skipping.", verbose)
            continue

    # Normal threshold for all digits
    if best_score < 0.5:  # Lower threshold since we're using resized images
        print_verbose(f"No confident match found (best score: {best_score:.2f})", verbose)
        return 0, best_score

    return best_match, best_score

def extract_digit_contour(image, min_area=5, edge_margin=1, verbose: bool = False):
    """Extract the main digit contour from an image, ignoring edge noise."""
    # Find all contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print_verbose("No contours found in image", verbose)
        return None, image, None
    
    # Debug: Print all contour areas
    print_verbose(f"Found {len(contours)} contours", verbose)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        print_verbose(f"Contour {i}: area={area:.1f}, bbox=({x},{y},{w},{h})", verbose)
    
    # Filter contours that are too close to the edges
    h, w = image.shape
    valid_contours = []
    for i, contour in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Debug: Print why contours are being filtered
        is_valid = True
        reasons = []
        if x <= edge_margin:
            reasons.append("too close to left edge")
            is_valid = False
        if y <= edge_margin:
            reasons.append("too close to top edge")
            is_valid = False
        if x + cw >= w - edge_margin:
            reasons.append("too close to right edge")
            is_valid = False
        if y + ch >= h - edge_margin:
            reasons.append("too close to bottom edge")
            is_valid = False
        if area <= min_area:
            reasons.append(f"area too small ({area:.1f} <= {min_area})")
            is_valid = False
            
        if is_valid:
            valid_contours.append(contour)
        else:
            print_verbose(f"Contour {i} rejected: {', '.join(reasons)}", verbose)
    
    if not valid_contours:
        print_verbose("No valid contours after filtering", verbose)
        # Debug: Save the image to see what we're working with
        # save_debug_step(image, "debug_no_valid_contours")
        return None, image, None
    
    # Get the largest valid contour
    main_contour = max(valid_contours, key=cv2.contourArea)
    print_verbose(f"Selected contour with area {cv2.contourArea(main_contour):.1f}", verbose)
    
    # Create a mask for the digit
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    
    # Apply the mask to get just the digit
    digit = cv2.bitwise_and(image, mask)
    
    # Create a contour image (just the contour, no fill)
    contour_img = np.zeros_like(image)
    cv2.drawContours(contour_img, [main_contour], -1, 255, 1)  # thickness=1 for just the outline
    
    # Debug: Save the intermediate steps
    # save_debug_step(mask, "debug_mask")
    # save_debug_step(digit, "debug_digit")
    # save_debug_step(contour_img, "debug_contour")
    
    return main_contour, digit, contour_img

def load_templates(template_dir='digits', target_size=(32, 32), verbose: bool = False):
    """Load digit templates and process them to match our digit format (white digit on black background)."""
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith('.png'):
            digit = int(filename[0])  # Get digit from filename (e.g., "1.png" -> 1)
            template_path = os.path.join(template_dir, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise ValueError(f"Could not read template at {template_path}")
            
            # Convert to binary (black digit on white background)
            _, binary = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print_verbose(f"Warning: No contours found in template {digit}", verbose)
                continue
                
            # Get the largest contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask for the digit (white where the digit is)
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [main_contour], -1, 255, -1)
            
            # Find the inner white region (the actual digit)
            inverted = cv2.bitwise_not(binary)
            inner_region = cv2.bitwise_and(inverted, mask)
            
            # Create a clean template (white digit on black background)
            template_img = np.zeros_like(binary)
            template_img[inner_region == 255] = 255
            
            # Optional: Erode slightly to remove thin outline
            kernel = np.ones((2,2), np.uint8)
            template_img = cv2.erode(template_img, kernel, iterations=1)
            
            # Crop and normalize the template
            template_img, digit_bounds = crop_and_normalize_digit(template_img, target_size)
            
            # Save template for debugging
            # save_debug_step(template_img, f"template_{digit}")
            
            templates[digit] = Digit(template_img, main_contour, None, None)
    
    return templates

def save_debug_step(image, step_name, output_dir='debug_steps'):
    """Save a debug image of an intermediate processing step."""
    os.makedirs(output_dir, exist_ok=True)
    if len(image.shape) == 2:
        # Convert grayscale to BGR for consistent saving
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{step_name}.png"), image)

def preprocess_image(image_path, target_size=(32, 32)):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # 1Save original image
    # save_debug_step(img, "1_original")
    
    # 2 Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # save_debug_step(gray, "2_grayscale")
    
    # 3 Apply thresholding to get black and white image
    # Note: We're using BINARY_INV so black digits are 255 (white) and background is 0 (black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # save_debug_step(binary, "3_binary")
    
    # 4Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Draw contours for debugging
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    # save_debug_step(contour_img, "4_contours")
    
    # Process each contour as a potential digit
    digits = []
    for i, contour in enumerate(contours):
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw bounding box for debugging
        bbox_img = img.copy()
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # save_debug_step(bbox_img, f"5_bbox_{i}")
        
        # Extract the digit region with some padding
        padding = 2
        digit_region = binary[max(0, y-padding):min(binary.shape[0], y+h+padding), 
                            max(0, x-padding):min(binary.shape[1], x+w+padding)]
        # save_debug_step(digit_region, f"6_digit_region_{i}")
        
        # Find contours within the digit region
        digit_contours, _ = cv2.findContours(digit_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not digit_contours:
            print(f"No contours found in digit {i}")
            continue
            
        # Find the largest contour in the digit region
        main_digit_contour = max(digit_contours, key=cv2.contourArea)
        
        # Create a mask for the digit (white where the digit is)
        digit_mask = np.zeros_like(digit_region)
        cv2.drawContours(digit_mask, [main_digit_contour], -1, 255, -1)
        # save_debug_step(digit_mask, f"7_digit_mask_{i}")
        
        # Find the inner white region (the actual digit)
        # First, invert the binary image so the digit is black (0) and background is white (255)
        inverted = cv2.bitwise_not(digit_region)
        # Then, use the mask to get only the inner region
        inner_region = cv2.bitwise_and(inverted, digit_mask)
        # save_debug_step(inner_region, f"8_inner_region_{i}")
        
        # Create a clean digit image
        # Start with a black background
        digit_img = np.zeros_like(digit_region)
        # Copy the inner region (which is white) to the black background
        digit_img[inner_region == 255] = 255
        
        # Optional: Erode slightly to remove thin outline
        kernel = np.ones((2,2), np.uint8)
        digit_img = cv2.erode(digit_img, kernel, iterations=1)
        
        # Crop and normalize the digit
        digit_img, digit_bounds = crop_and_normalize_digit(digit_img, target_size)
        
        # Create a visualization of the contour
        contour_vis = digit_img.copy()
        cv2.drawContours(contour_vis, [main_digit_contour], -1, 128, 1)  # Draw contour in gray
        # save_debug_step(contour_vis, f"9_digit_contour_{i}")
        
        # Save the final digit image
        # save_debug_step(digit_img, f"10_digit_final_{i}")
        
        # Update bbox to reflect the actual digit bounds
        x += digit_bounds[0]
        y += digit_bounds[1]
        w = digit_bounds[2]
        h = digit_bounds[3]
        
        digits.append(Digit(digit_img, main_digit_contour, None, (x, y, w, h)))
    
    return digits

def recognize_digit_ocr(digit_img, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'):
    """
    Recognize a digit using Tesseract OCR.
    Uses PSM 10 (treat as single character) and whitelists only digits.
    
    Args:
        digit_img: Binary image of the digit (white digit on black background)
        config: Tesseract configuration string
        
    Returns:
        tuple: (recognized_digit, confidence)
    """
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(digit_img)
        
        # Get OCR results with confidence scores
        data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
        
        # Filter out empty results and get the best confidence
        confidences = [float(conf) for conf, text in zip(data['conf'], data['text']) if text.strip()]
        if not confidences:
            print("No OCR results found")
            return 0, 0.0
            
        best_conf = max(confidences)
        best_idx = confidences.index(best_conf)
        recognized_digit = int(data['text'][best_idx])
        
        # Normalize confidence to 0-1 range (Tesseract gives 0-100)
        confidence = best_conf / 100.0
        
        print(f"OCR recognized: {recognized_digit} with confidence {confidence:.2f}")
        return recognized_digit, confidence
        
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return 0, 0.0

def process_digits(digits: List[Digit], templates: Dict[int, Digit], method='template', save_debug=True, verbose: bool = False):
    """
    Process all found digits and return their classifications.
    
    Args:
        digits: List of Digit objects to process
        templates: Dictionary of template Digit objects
        method: One of 'template', 'ocr', or 'pixel' for different matching methods
        save_debug: Whether to save debug images
    """
    results = []
    
    for i, digit_data in enumerate(digits):
        print_verbose(f"\nProcessing digit {i}:", verbose)
        print_verbose(f"Position: {digit_data.bbox}", verbose)
        
        if digit_data.image is None:
            print_verbose(f"Warning: No image found for digit {i}", verbose)
            continue
        
        if method == 'ocr':
            # Use OCR for recognition
            digit, confidence = recognize_digit_ocr(digit_data.image)
        elif method == 'pixel':
            # Use pixel-wise comparison
            digit, confidence = find_most_similar_template(i, digit_data.image, templates)
        else:
            # Use template matching
            digit, confidence = find_matching_template(digit_data.image, templates)
        
        # Save debug image if requested
        if save_debug:
            x, y, w, h = digit_data.bbox
            # save_debug_image(digit_data.image, digit, confidence, y, x)
        
        results.append({
            'digit': digit,
            'confidence': confidence,
            'position': digit_data.bbox,
            'method': method
        })
    
    return results

def save_debug_image(cell, digit, confidence, row, col, output_dir='debug_samples'):
    """Save a debug image of the cell with its classification."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert cell to 3 channels for colored text
    if len(cell.shape) == 2:
        cell_color = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
    else:
        cell_color = cell.copy()
    
    # Add text showing the digit and confidence
    text = f"{digit} ({confidence:.2f})"
    cv2.putText(cell_color, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 1)
    
    # Save the image
    filename = f"cell_r{row}_c{col}_d{digit}_conf{confidence:.2f}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cell_color)

def process_image_to_grid(image_path: Optional[str] = None, 
                         method: str = 'template', 
                         templates: Optional[Dict] = None,
                         screenshot_region: Optional[Tuple[int, int, int, int]] = None,
                         verbose: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    """
    Process an image and return the extracted digit grid.
    
    Args:
        image_path: Path to the input image, or None to take a screenshot
        method: One of 'template', 'ocr', or 'pixel' for different matching methods
        templates: Optional pre-loaded templates dictionary. If None, will load templates.
        screenshot_region: Optional (left, top, width, height) tuple to capture a specific region
        
    Returns:
        tuple: (grid, results) where:
            - grid is a numpy array of shape (10, 17) containing the extracted digits
            - results is a list of dictionaries containing detailed results for each digit
    """
    try:
        # Take screenshot if no image path provided
        if image_path is None:
            from clicker import take_screenshot
            print_verbose("Taking screenshot...", verbose)
            image_path = take_screenshot(screenshot_region)
        
        # Load templates if not provided
        if templates is None and method in ['template', 'pixel']:
            print_verbose("Loading digit templates...", verbose)
            templates = load_templates()
        elif method not in ['template', 'pixel']:
            templates = {}
        
        # Preprocess the image and find digits
        print_verbose("Processing image...", verbose)
        digits = preprocess_image(image_path)
        print_verbose(f"Found {len(digits)} potential digits", verbose)
        
        # Process all digits
        print_verbose(f"Matching digits using {method} method...", verbose)
        results = process_digits(digits, templates, method=method, save_debug=True)
        
        # Group digits into rows based on y-coordinates
        y_coords = sorted(set(r['position'][1] for r in results))
        
        # Calculate average height for threshold calculations
        avg_height = sum(r['position'][3] for r in results) / len(results)
        
        # If we have more than 10 unique y-coordinates, we need to cluster them
        if len(y_coords) > 10:
            y_threshold = avg_height * 0.5
            
            # Group y-coordinates that are close together
            row_y_coords = []
            current_group = [y_coords[0]]
            
            for y in y_coords[1:]:
                if y - current_group[-1] <= y_threshold:
                    current_group.append(y)
                else:
                    row_y_coords.append(sum(current_group) / len(current_group))
                    current_group = [y]
            
            if current_group:
                row_y_coords.append(sum(current_group) / len(current_group))
            
            # Ensure we have exactly 10 rows
            if len(row_y_coords) > 10:
                while len(row_y_coords) > 10:
                    diffs = [(i, row_y_coords[i+1] - row_y_coords[i]) 
                            for i in range(len(row_y_coords)-1)]
                    merge_idx = min(diffs, key=lambda x: x[1])[0]
                    row_y_coords[merge_idx] = (row_y_coords[merge_idx] + row_y_coords[merge_idx+1]) / 2
                    row_y_coords.pop(merge_idx + 1)
        else:
            row_y_coords = y_coords
        
        # Sort row_y_coords to ensure they're in order
        row_y_coords.sort()
        
        # Create and fill the grid
        grid = np.zeros((10, 17), dtype=int)  # 10 rows, 17 columns
        
        # For each row, find digits that belong to it and sort by x-coordinate
        for row_idx, row_y in enumerate(row_y_coords):
            # Find digits in this row (within threshold of row_y)
            row_digits = []
            for result in results:
                y = result['position'][1]
                if abs(y - row_y) <= avg_height * 0.5:
                    row_digits.append(result)
            
            # Sort digits in this row by x-coordinate
            row_digits.sort(key=lambda r: r['position'][0])
            
            # Fill the grid row
            for col_idx, digit in enumerate(row_digits[:17]):
                grid[row_idx, col_idx] = digit['digit']
        
        return grid, results
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def print_grid(grid):
   # Print the grid
    print("\nExtracted Grid (17x10):")
    print("-" * 35)
    for row in grid:
        row_str = " ".join(str(d) for d in row)
        print(row_str)
    print("-" * 35)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_grid.py <image_path> [--method template|ocr|pixel]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    method = 'template'  # default method
    
    # Parse method argument
    if len(sys.argv) > 2:
        if sys.argv[2] == '--ocr':
            method = 'ocr'
        elif sys.argv[2] == '--pixel':
            method = 'pixel'
        elif sys.argv[2] == '--template':
            method = 'template'
        else:
            print("Unknown method. Using default template matching.")
    
    try:
        # Process the image
        grid, results = process_image_to_grid(image_path, method)
        
        # Print individual results
        print("\nExtracted Digits:")
        print("-" * 50)
        for i, result in enumerate(results):
            x, y, w, h = result['position']
            print(f"Digit {i}: {result['digit']} (confidence: {result['confidence']:.2f}) "
                  f"at position ({x}, {y}) using {result['method']}")
        print("-" * 50)
        
        print_grid(grid)
        
        print("\nDebug images have been saved to the 'debug_steps' directory")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 