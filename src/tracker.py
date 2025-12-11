"""
Centroid Tracker
Tracks objects (people) across video frames using centroids
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class CentroidTracker:
    """
    Track objects using centroids
    Based on the simple centroid tracking algorithm
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        """
        Initialize Centroid Tracker
        
        Args:
            max_disappeared: Maximum frames an object can disappear before deletion
            max_distance: Maximum distance for matching centroids (pixels)
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # {object_id: (cx, cy)}
        self.disappeared = OrderedDict()  # {object_id: num_frames_disappeared}
        self.bbox = OrderedDict()  # {object_id: (x1, y1, x2, y2)}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Track crossing events
        self.crossed_line = set()  # Set of object_ids that crossed entry line
        self.crossing_direction = OrderedDict()  # {object_id: 'left_to_right' or 'right_to_left'}
        
        logger.info(f"✅ Centroid Tracker initialized (max_disappeared={max_disappeared}, max_distance={max_distance})")
    
    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]):
        """
        Register a new object
        
        Args:
            centroid: (cx, cy) coordinates
            bbox: (x1, y1, x2, y2) bounding box
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.bbox[self.next_object_id] = bbox
        self.crossing_direction[self.next_object_id] = None
        
        logger.debug(f"Registered new object ID {self.next_object_id}")
        self.next_object_id += 1
    
    def deregister(self, object_id: int):
        """
        Deregister (remove) an object
        
        Args:
            object_id: ID of object to remove
        """
        logger.debug(f"Deregistering object ID {object_id}")
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bbox[object_id]
        if object_id in self.crossed_line:
            self.crossed_line.remove(object_id)
        if object_id in self.crossing_direction:
            del self.crossing_direction[object_id]
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes
        
        Returns:
            Dictionary of {object_id: (x1, y1, x2, y2)}
        """
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.bbox
        
        # Calculate centroids from detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detections[i])
        
        # Otherwise, match existing objects with new detections
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance for each existing object
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Match objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is within threshold
                if D[row, col] > self.max_distance:
                    continue
                
                # Update object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bbox[object_id] = detections[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], detections[col])
        
        return self.bbox
    
    def check_line_crossing(self, line_x: int, line_buffer: int = 20) -> int:
        """
        Check if any tracked objects crossed the entry line
        
        Args:
            line_x: X coordinate of entry line
            line_buffer: Buffer zone around line (pixels)
        
        Returns:
            Number of new crossings in this frame
        """
        new_crossings = 0
        
        for object_id, centroid in self.objects.items():
            cx, cy = centroid
            
            # Get previous direction if exists
            prev_direction = self.crossing_direction.get(object_id)
            
            # Determine current position relative to line
            if cx < (line_x - line_buffer):
                current_direction = 'left'
            elif cx > (line_x + line_buffer):
                current_direction = 'right'
            else:
                current_direction = 'on_line'
            
            # Check for crossing
            if prev_direction and current_direction != 'on_line':
                # Crossed from left to right
                if prev_direction == 'left' and current_direction == 'right':
                    if object_id not in self.crossed_line:
                        self.crossed_line.add(object_id)
                        new_crossings += 1
                        logger.info(f"Object {object_id} crossed line (left→right)")
                
                # Crossed from right to left
                elif prev_direction == 'right' and current_direction == 'left':
                    if object_id not in self.crossed_line:
                        self.crossed_line.add(object_id)
                        new_crossings += 1
                        logger.info(f"Object {object_id} crossed line (right→left)")
            
            # Update direction
            if current_direction != 'on_line':
                self.crossing_direction[object_id] = current_direction
        
        return new_crossings
    
    def get_total_crossings(self) -> int:
        """Get total number of objects that crossed the line"""
        return len(self.crossed_line)
    
    def reset(self):
        """Reset tracker (clear all tracked objects)"""
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()
        self.crossed_line = set()
        self.crossing_direction = OrderedDict()
        logger.info("Tracker reset")

if __name__ == "__main__":
    # Test centroid tracker
    import sys
    sys.path.append('..')
    from src.utils import setup_logging
    import cv2
    
    setup_logging(level="DEBUG")
    
    # Create tracker
    tracker = CentroidTracker(max_disappeared=30, max_distance=50)
    
    # Simulate detections
    print("Simulating person tracking...")
    
    # Frame 1: Person on left side
    detections1 = [(100, 200, 150, 300)]
    objects = tracker.update(detections1)
    print(f"Frame 1: {len(objects)} tracked objects")
    
    # Frame 2: Person moves right
    detections2 = [(200, 200, 250, 300)]
    objects = tracker.update(detections2)
    crossings = tracker.check_line_crossing(line_x=300)
    print(f"Frame 2: {len(objects)} tracked objects, {crossings} new crossings")
    
    # Frame 3: Person crosses line
    detections3 = [(350, 200, 400, 300)]
    objects = tracker.update(detections3)
    crossings = tracker.check_line_crossing(line_x=300)
    print(f"Frame 3: {len(objects)} tracked objects, {crossings} new crossings")
    
    # Frame 4: Another person appears
    detections4 = [(350, 200, 400, 300), (50, 100, 100, 200)]
    objects = tracker.update(detections4)
    crossings = tracker.check_line_crossing(line_x=300)
    print(f"Frame 4: {len(objects)} tracked objects, {crossings} new crossings")
    
    print(f"\nTotal crossings: {tracker.get_total_crossings()}")
    print("✅ Centroid tracker test completed!")
