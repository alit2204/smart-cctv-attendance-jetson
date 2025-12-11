"""
Face Detector using CenterFace
Detects faces in images and returns bounding boxes
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    CenterFace face detector
    Anchor-free single-stage face detection
    """
    
    def __init__(self, model_path: str, 
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.3,
                 input_size: Tuple[int, int] = (640, 480)):
        """
        Initialize Face Detector
        
        Args:
            model_path: Path to centerface.onnx model
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS threshold for filtering overlapping boxes
            input_size: Model input size (width, height)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Load ONNX model
        logger.info(f"Loading CenterFace model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"✅ CenterFace loaded successfully")
        logger.info(f"   Input: {self.input_name}, Outputs: {self.output_names}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CenterFace
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Preprocessed image for model input
        """
        # Resize to input size
        resized = cv2.resize(image, self.input_size)
        
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
        Postprocess model outputs to get bounding boxes
        
        Args:
            outputs: Model outputs
            original_shape: Original image shape (height, width)
        
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        # CenterFace outputs: [heatmap, scale, offset, landmarks]
        # We'll use heatmap and scale to get bounding boxes
        
        heatmap = outputs[0][0]  # (1, H, W)
        scale = outputs[1][0]    # (2, H, W) - width, height scales
        
        # Get detections from heatmap
        detections = []
        
        # Find peaks in heatmap (face centers)
        threshold_map = (heatmap > self.confidence_threshold).astype(np.uint8)
        
        for c in range(heatmap.shape[0]):  # For each channel (usually 1 for face)
            peaks = self._find_peaks(heatmap[c], threshold_map[c])
            
            for y, x in peaks:
                confidence = float(heatmap[c, y, x])
                
                # Get scale (bbox size)
                w_scale = float(scale[0, y, x])
                h_scale = float(scale[1, y, x])
                
                # Calculate bbox coordinates
                bbox_w = w_scale * self.input_size[0]
                bbox_h = h_scale * self.input_size[1]
                
                x1 = int(x * (original_shape[1] / self.input_size[0]) - bbox_w / 2)
                y1 = int(y * (original_shape[0] / self.input_size[1]) - bbox_h / 2)
                x2 = int(x1 + bbox_w)
                y2 = int(y1 + bbox_h)
                
                # Clip to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_shape[1], x2)
                y2 = min(original_shape[0], y2)
                
                detections.append((x1, y1, x2, y2, confidence))
        
        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections)
        
        return detections
    
    def _find_peaks(self, heatmap: np.ndarray, threshold_map: np.ndarray) -> List[Tuple[int, int]]:
        """Find local peaks in heatmap"""
        from scipy.ndimage import maximum_filter
        
        # Apply maximum filter to find local maxima
        local_max = maximum_filter(heatmap, size=3) == heatmap
        
        # Combine with threshold mask
        peaks = local_max & threshold_map.astype(bool)
        
        # Get peak coordinates
        y_coords, x_coords = np.where(peaks)
        
        return list(zip(y_coords, x_coords))
    
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
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of (x1, y1, x2, y2, confidence) face detections
        """
        original_shape = image.shape[:2]  # (height, width)
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, original_shape)
        
        logger.debug(f"Detected {len(detections)} faces")
        
        return detections

if __name__ == "__main__":
    # Test face detector
    import sys
    sys.path.append('..')
    from src.utils import setup_logging
    
    setup_logging(level="DEBUG")
    
    # Initialize detector
    detector = FaceDetector(
        model_path="../models/centerface.onnx",
        confidence_threshold=0.5
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Draw bounding boxes
        for x1, y1, x2, y2, conf in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Face detector test completed!")
