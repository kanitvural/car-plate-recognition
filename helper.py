import cv2
import numpy as np
import logging
from ultralytics import YOLO
from typing import Tuple, List

# Logging yapılandırması
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def detect_plate(image: np.ndarray, model_path: str) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """
    Detects one or more license plates in an image using a YOLOv8 model.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        model_path (str): Path to the YOLOv8 model file.

    Returns:
        Tuple[np.ndarray, List[np.ndarray], int]: 
            - Image with bounding boxes drawn (np.ndarray).
            - List of cropped plate images (List[np.ndarray]).
            - Number of detected plates.
    """

    logging.info("Loading image...")

    # Convert image to a writable NumPy array
    image_array = np.asarray(image).copy()

    # Load YOLOv8 model
    model = YOLO(model_path)

    # Perform object detection
    logging.info("Detecting plates...")
    results = model(image_array)[0]

    # Get detection results
    detected_boxes = results.boxes.data.tolist()
    is_detected = len(detected_boxes)
    cropped_images = []

    if is_detected > 0:
        logging.info(f"{is_detected} plate(s) detected.")
        threshold = 0.5

        for result in detected_boxes:
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if score >= threshold:
                cropped_images.append(image_array[y1:y2, x1:x2])  # Append cropped plate

                # Draw bounding box and label
                class_name = results.names[class_id]
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {score * 100:.2f}%"
                cv2.putText(image_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 1, cv2.LINE_AA)

    else:
        logging.info("No plate detected.")
        
        text = "No plate detected."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2 
        thickness = 3
                
        img_height, img_width, _ = image_array.shape
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (img_width - text_width) // 2
        y = (img_height + text_height) // 2

        black_image = np.zeros_like(image_array, dtype=np.uint8)
        
        image_array = black_image
        cropped_images.append(black_image)
        cv2.putText(image_array, text, (x, y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

    return image_array, cropped_images, is_detected




 # Modelinizi doğru şekilde içeri aktarın

def read_plate(image: np.ndarray, model_path: str) -> Tuple[np.ndarray, str]:
    """
    Reads the text from a cropped license plate image using a YOLOv8 model.

    Args:
        image (np.ndarray): Cropped license plate image.
        model_path (str): Path to the plate reading model.

    Returns:
        Tuple[np.ndarray, str]: 
            - Image with plate bounding boxes (np.ndarray).
            - Text read from the plate (str).
    """

    logging.info("Reading plate...")

    # Convert image to a NumPy array
    image_array = np.asarray(image)

    # Load YOLOv8 model for reading the plate
    model = YOLO(model_path)
    
    # Perform text recognition
    results = model(image_array)[0]
    
    is_detected = len(results.boxes.data.tolist())
    detected_text = ""
    class_names = []  # This will store all detected characters
    coordinates = []  # To store the coordinates (x1) for sorting
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 
    thickness = 1

    if is_detected > 0:
        threshold = 0.5
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if score > threshold:
                # Draw bounding box and label
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                class_name = results.names[class_id]
                class_names.append(class_name)  # Append detected character
                coordinates.append(x1)  # Store the x1 coordinate for sorting
                
                # Write the class name on the image
                cv2.putText(image_array, class_name, (x1 + 10, y1 + 30), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # Sort the characters based on their x1 coordinate (left-to-right order)
        sorted_indices = np.argsort(coordinates)
        sorted_class_names = [class_names[i] for i in sorted_indices]

        # Combine all detected characters to form the full plate text
        detected_text = ''.join(sorted_class_names)  # Combine class names into a single text string

    return image_array, detected_text
