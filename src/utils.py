"""
Utility Functions for Smart CCTV Attendance System
Author: Umbreen, Ali, Hamza
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import time

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_file: str = "logs/system.log", level: str = "INFO"):
    """Setup logging configuration"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: Input image (BGR format)
        target_size: (width, height) tuple
    
    Returns:
        Preprocessed image
    """
    # Resize
    resized = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    return normalized

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Normalize embeddings
    embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-6)
    embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-6)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    return float(similarity)

def draw_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], 
              label: str, color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image
    
    Args:
        image: Input image
        bbox: (x1, y1, x2, y2) coordinates
        label: Text label
        color: Box color in BGR
        thickness: Line thickness
    
    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    y1_label = max(y1, label_size[1] + 10)
    cv2.rectangle(image, (x1, y1_label - label_size[1] - 10), 
                  (x1 + label_size[0], y1_label), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1, y1_label - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return image

def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
    
    Returns:
        IoU score (0 to 1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def non_max_suppression(boxes: List[Tuple], confidences: List[float], 
                        threshold: float = 0.3) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    
    Args:
        boxes: List of (x1, y1, x2, y2) boxes
        confidences: List of confidence scores
        threshold: NMS threshold
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    # Get indices sorted by confidence
    indices = np.argsort(confidences)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep highest confidence box
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = np.array([calculate_iou(boxes[current], boxes[idx]) 
                        for idx in indices[1:]])
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][ious < threshold]
    
    return keep

def calculate_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate centroid of bounding box
    
    Args:
        bbox: (x1, y1, x2, y2)
    
    Returns:
        (cx, cy) centroid coordinates
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def euclidean_distance(point1: Tuple[int, int], 
                       point2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class FPSCounter:
    """Simple FPS counter"""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self):
        """Update FPS counter"""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])

def create_directory_structure():
    """Create necessary directories if they don't exist"""
    directories = [
        "config",
        "models",
        "database",
        "student_photos",
        "logs",
        "src"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created successfully!")

if __name__ == "__main__":
    # Test functions
    print("Testing utility functions...")
    
    # Test config loading
    config = load_config()
    print(f"✅ Config loaded: {len(config)} sections")
    
    # Test cosine similarity
    emb1 = np.random.rand(128)
    emb2 = np.random.rand(128)
    sim = cosine_similarity(emb1, emb2)
    print(f"✅ Cosine similarity: {sim:.3f}")
    
    # Test FPS counter
    fps_counter = FPSCounter()
    for _ in range(10):
        fps_counter.update()
        time.sleep(0.03)
    print(f"✅ FPS counter: {fps_counter.get_fps():.1f} FPS")
    
    print("\n✅ All utility functions working!")
