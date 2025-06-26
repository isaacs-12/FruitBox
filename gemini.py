import cv2
import numpy as np
import pytesseract
from PIL import Image

from ocr_grid import save_debug_step

def extract_digits_from_grid(image_path):
    """
    Extracts digits from a grid-like image and returns them as a NumPy array.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        np.ndarray: A 2D NumPy array containing the extracted digits, or None if
                    the process fails.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to get a binary image
        # This helps separate digits from the background more effectively
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming a relatively uniform grid, we can sort contours by their y-coordinate
        # to group them by row, and then by x-coordinate for columns within each row.
        # This part might need fine-tuning based on actual image alignment.
        digit_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter out very small or very large contours that are unlikely to be digits
            if 10 < w < 70 and 10 < h < 70: # These values are estimations, adjust as needed
                digit_boxes.append((x, y, w, h))

        # Sort boxes primarily by y-coordinate (rows) and then by x-coordinate (columns)
        # We'll need a bit more sophisticated sorting to handle the grid correctly.
        # A simple sort by y then x assumes perfect alignment.
        # A better approach for a grid is to group by proximity in Y, then sort by X within groups.

        # Let's try to infer grid structure more robustly
        # Estimate average height and width of a digit/apple
        if not digit_boxes:
            print("No significant contours found that could be digits.")
            return None

        avg_h = np.mean([box[3] for box in digit_boxes])
        avg_w = np.mean([box[2] for box in digit_boxes])

        # Group bounding boxes into rows
        rows = []
        digit_boxes.sort(key=lambda b: b[1]) # Sort by Y-coordinate
        current_row = []
        if digit_boxes:
            current_y_threshold = digit_boxes[0][1] + avg_h / 2 # Threshold for a new row
            for box in digit_boxes:
                if box[1] < current_y_threshold:
                    current_row.append(box)
                else:
                    rows.append(sorted(current_row, key=lambda b: b[0])) # Sort by X within row
                    current_row = [box]
                    current_y_threshold = box[1] + avg_h / 2
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b[0]))


        extracted_digits = []
        for row_boxes in rows:
            row_digits = []
            for (x, y, w, h) in row_boxes:
                # Crop the digit from the original grayscale image or the thresholded image
                digit_roi = gray[y:y+h, x:x+w]

                # Resize for better OCR performance (optional, depends on digit size)
                # digit_roi = cv2.resize(digit_roi, (50, 50), interpolation=cv2.INTER_AREA)

                # Add padding to the ROI, sometimes helps Tesseract
                padded_digit_roi = cv2.copyMakeBorder(digit_roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

                save_debug_step(padded_digit_roi, f"padded_digit_roi_{x}_{y}_{w}_{h}")

                # Use Tesseract to recognize the digit
                # config='--psm 10' for single digit recognition
                # config='--psm 6' for a single uniform block of text
                # -c tessedit_char_whitelist=0123456789 to restrict to digits
                text = pytesseract.image_to_string(padded_digit_roi,
                                                  config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()

                # Clean up the output - Tesseract might return empty string or non-digits
                if text.isdigit() and len(text) == 1:
                    row_digits.append(int(text))
                else:
                    # If Tesseract fails, try using the thresholded image, or try different PSMs
                    text_thresh = pytesseract.image_to_string(thresh[y:y+h, x:x+w],
                                                               config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()
                    if text_thresh.isdigit() and len(text_thresh) == 1:
                        row_digits.append(int(text_thresh))
                    else:
                        print(f"Warning: Could not recognize digit at ({x},{y},{w},{h}). Recognized as: '{text}' or '{text_thresh}'. Using 0.")
                        row_digits.append(0) # Placeholder for unrecognized digits

            extracted_digits.append(row_digits)

        # Convert to NumPy array
        # Ensure all rows have the same number of columns, pad with 0 if necessary
        max_cols = 0
        if extracted_digits:
            max_cols = max(len(row) for row in extracted_digits)

        padded_digits = []
        for row in extracted_digits:
            padded_row = row + [0] * (max_cols - len(row))
            padded_digits.append(padded_row)

        return np.array(padded_digits)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- How to use the script ---
if __name__ == "__main__":
    image_file = "test_data/test_2.png" # Replace with your image file name

    digit_grid = extract_digits_from_grid(image_file)

    if digit_grid is not None:
        print("Extracted Digits (NumPy Array):")
        print(digit_grid)
        print("\nShape of the array:", digit_grid.shape)
    else:
        print("Digit extraction failed.")