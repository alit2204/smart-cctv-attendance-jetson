"""
Face Recognizer using MobileFaceNet
Extracts 128-D embeddings from face images for recognition
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class MobileFaceNet(nn.Module):
    """MobileFaceNet architecture"""
    
    def __init__(self, embedding_size=128):
        super(MobileFaceNet, self).__init__()
        
        # Architecture will be loaded from pretrained weights
        # This is a simplified placeholder
        self.embedding_size = embedding_size
    
    def forward(self, x):
        # Forward pass defined by loaded weights
        return x

class FaceRecognizer:
    """
    Face Recognizer using MobileFaceNet
    Extracts face embeddings for recognition
    """
    
    def __init__(self, model_path: str, 
                 face_size: Tuple[int, int] = (112, 112),
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Face Recognizer
        
        Args:
            model_path: Path to mobilefacenet.pth model
            face_size: Input face size (width, height)
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.face_size = face_size
        self.device = device
        
        # Load model
        logger.info(f"Loading MobileFaceNet model from {model_path}")
        logger.info(f"Using device: {device}")
        
        try:
            # Load pretrained weights
            self.model = self._load_model(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ MobileFaceNet loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_model(self, model_path: str):
        """Load MobileFaceNet model from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            # Checkpoint is the model itself
            return checkpoint
        
        # Create model and load weights
        model = self._build_mobilefacenet()
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def _build_mobilefacenet(self):
        """Build MobileFaceNet architecture"""
        # Simplified MobileFaceNet architecture
        # In practice, this should match the exact architecture used during training
        
        class Bottleneck(nn.Module):
            def __init__(self, inp, oup, stride, expansion):
                super(Bottleneck, self).__init__()
                self.connect = stride == 1 and inp == oup
                
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(inp * expansion),
                    nn.PReLU(inp * expansion),
                    
                    nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, 
                             groups=inp * expansion, bias=False),
                    nn.BatchNorm2d(inp * expansion),
                    nn.PReLU(inp * expansion),
                    
                    nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            
            def forward(self, x):
                if self.connect:
                    return x + self.conv(x)
                else:
                    return self.conv(x)
        
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            
            # Add bottleneck layers
            Bottleneck(64, 64, 2, 2),
            Bottleneck(64, 128, 2, 4),
            Bottleneck(128, 128, 1, 2),
            
            nn.Conv2d(128, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128)
        )
        
        return model
    
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for MobileFaceNet
        
        Args:
            face_image: Face image (BGR format)
        
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        resized = cv2.resize(face_image, self.face_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] (MobileFaceNet standard)
        normalized = (rgb.astype(np.float32) - 127.5) / 128.0
        
        # Transpose to (C, H, W) and add batch dimension
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 128-D embedding from face image
        
        Args:
            face_image: Face image (BGR format)
        
        Returns:
            128-D embedding vector (numpy array)
        """
        # Preprocess
        input_tensor = self.preprocess(face_image)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Convert to numpy and normalize
        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        logger.debug(f"Extracted embedding shape: {embedding.shape}")
        
        return embedding
    
    def compare_embeddings(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        # Normalize embeddings
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-6)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-6)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def identify(self, face_image: np.ndarray, 
                database_embeddings: dict,
                threshold: float = 0.6) -> Tuple[str, float]:
        """
        Identify face by comparing with database embeddings
        
        Args:
            face_image: Face image to identify
            database_embeddings: Dict of {student_id: embedding}
            threshold: Minimum similarity threshold
        
        Returns:
            (student_id, similarity) or (None, 0.0) if no match
        """
        # Extract embedding from input face
        query_embedding = self.extract_embedding(face_image)
        
        # Compare with all database embeddings
        best_match = None
        best_similarity = 0.0
        
        for student_id, db_embedding in database_embeddings.items():
            similarity = self.compare_embeddings(query_embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = student_id
        
        # Check if best match meets threshold
        if best_similarity >= threshold:
            logger.debug(f"Identified: {best_match} (similarity: {best_similarity:.3f})")
            return best_match, best_similarity
        else:
            logger.debug(f"No match found (best: {best_similarity:.3f})")
            return None, 0.0

if __name__ == "__main__":
    # Test face recognizer
    import sys
    sys.path.append('..')
    from src.utils import setup_logging
    from src.face_detector import FaceDetector
    
    setup_logging(level="DEBUG")
    
    # Initialize detector and recognizer
    detector = FaceDetector(
        model_path="../models/centerface.onnx",
        confidence_threshold=0.5
    )
    
    recognizer = FaceRecognizer(
        model_path="../models/mobilefacenet.pth"
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save embedding")
    saved_embedding = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Process first face
        if len(faces) > 0:
            x1, y1, x2, y2, conf = faces[0]
            
            # Extract face
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                # Extract embedding
                embedding = recognizer.extract_embedding(face_img)
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # If we have saved embedding, compare
                if saved_embedding is not None:
                    similarity = recognizer.compare_embeddings(embedding, saved_embedding)
                    label = f"Similarity: {similarity:.3f}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            x1, y1, x2, y2, _ = faces[0]
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                saved_embedding = recognizer.extract_embedding(face_img)
                print("✅ Embedding saved! Show your face again to compare.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Face recognizer test completed!")
