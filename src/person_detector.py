"""
Person Detector using SSD MobileNet
Detects people in images for tailgating detection
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class PersonDetector:
    """
    Person Detector using SSD MobileNet
    Detects people in video frames for tailgating monitoring
    """
    
    def __init__(self, model_path: str,
                 confidence_threshold: float = 0.4,
                 nms_threshold: float = 0.3,
                 person_class_id: int = 1):
        """
        Initialize Person Detector
        
        Args:
            model_path: Path to ssd_mobilenet.onnx model
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS threshold
            person_class_id: Class ID for person in COCO dataset (default: 1)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.person_class_id = person_class_id
        
        # Load ONNX model
        logger.info(f"Loading SSD MobileNet model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"✅ SSD MobileNet loaded successfully")
        logger.info(f"   Input: {self.input_name}")
        logger.info(f"   Outputs: {len(self.output_names)} outputs")
    
    def preprocess(self, image: np.ndarray, 
                   input_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
        """
        Preprocess image for SSD MobileNet
        
        Args:
            image: Input image (BGR format)
            input_size: Model input size (default: 300x300)
        
        Returns:
            Preprocessed image tensor
        """
        # Resize
        resized = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (1, 3, H, W) - NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def postprocess(self, outputs: List[np.ndarray],
                    original_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float]]:
        """
        Postprocess model outputs to get person bounding boxes
        
        Args:
            outputs: Model outputs [boxes, scores, classes]
            original_shape: Original image shape (height, width)
        
        Returns:
            List of (x1, y1, x2, y2, confidence) for detected persons
        """
        # SSD MobileNet outputs format varies, handle both common formats
        
        if len(outputs) >= 3:
            # Format 1: [boxes, classes, scores, num_detections]
            boxes = outputs[0][0]  # (N, 4) normalized coordinates
            classes = outputs[1][0]  # (N,) class ids
            scores = outputs[2][0]  # (N,) confidence scores
        else:
            # Format 2: Combined output
            # Will need to parse based on specific model format
            logger.warning("Unexpected output format, using fallback parsing")
            return []
        
        detections = []
        height, width = original_shape
        
        for box, class_id, score in zip(boxes, classes, scores):
            # Filter by class (person) and confidence
            if int(class_id) != self.person_class_id:
                continue
            
            if score < self.confidence_threshold:
                continue
            
            # Convert normalized coordinates to pixel coordinates
            # Box format: [ymin, xmin, ymax, xmax] (normalized 0-1)
            y1 = int(box[0] * height)
            x1 = int(box[1] * width)
            y2 = int(box[2] * height)
            x2 = int(box[3] * width)
            
            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            detections.append((x1, y1, x2, y2, float(score)))
        
        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float]]:
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Convert to numpy arrays
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confidences = np.array([d[4] for d in detections])
        
        # Sort by confidence
        indices = np.argsort(confidences)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]
            
            ious = self._calculate_iou_batch(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            indices = indices[1:][ious < self.nms_threshold]
        
        return [detections[i] for i in keep]
    
    def _calculate_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area_box + area_boxes - intersection
        
        return intersection / (union + 1e-6)
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect people in image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of (x1, y1, x2, y2, confidence) person detections
        """
        original_shape = image.shape[:2]  # (height, width)
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, original_shape)
        
        logger.debug(f"Detected {len(detections)} people")
        
        return detections
    
    def count_people_in_zone(self, detections: List[Tuple[int, int, int, int, float]],
                            zone_x: int, zone_width: int) -> int:
        """
        Count people in a specific zone (for entry detection)
        
        Args:
            detections: List of person detections
            zone_x: X coordinate of zone center
            zone_width: Width of zone
        
        Returns:
            Number of people in zone
        """
        count = 0
        zone_x1 = zone_x - zone_width // 2
        zone_x2 = zone_x + zone_width // 2
        
        for x1, y1, x2, y2, conf in detections:
            # Calculate person center
            person_center_x = (x1 + x2) // 2
            
            # Check if person center is in zone
            if zone_x1 <= person_center_x <= zone_x2:
                count += 1
        
        return count

if __name__ == "__main__":
    # Test person detector
    import sys
    sys.path.append('..')
    from src.utils import setup_logging, FPSCounter
    
    setup_logging(level="DEBUG")
    
    # Initialize detector
    detector = PersonDetector(
        model_path="../models/ssd_mobilenet.onnx",
        confidence_threshold=0.4
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    fps_counter = FPSCounter()
    
    print("Press 'q' to quit")
    
    # Draw entry line
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    entry_line_x = frame_width // 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect people
        people = detector.detect(frame)
        
        # Draw entry line
        cv2.line(frame, (entry_line_x, 0), (entry_line_x, frame.shape[0]),
                (255, 0, 0), 2)
        
        # Draw bounding boxes
        for x1, y1, x2, y2, conf in people:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Count people in entry zone
        zone_count = detector.count_people_in_zone(people, entry_line_x, 100)
        
        # Display info
        fps_counter.update()
        info_text = f"People: {len(people)} | In Zone: {zone_count} | FPS: {fps_counter.get_fps():.1f}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Person Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Person detector test completed!")
