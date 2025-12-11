"""
Integrated Smart CCTV Attendance System with Tailgating Detection
Combines face recognition attendance with person tracking for security
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.append('.')
from src import (FaceDetector, FaceRecognizer, PersonDetector, 
                CentroidTracker, DatabaseManager)
from src.utils import load_config, setup_logging, FPSCounter, draw_bbox

class SmartCCTVSystem:
    """
    Integrated attendance and tailgating detection system
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the complete system"""
        self.config = load_config(config_path)
        self.logger = setup_logging(
            log_file=self.config['logging']['log_file'],
            level=self.config['logging']['level']
        )
        
        self.logger.info("="*70)
        self.logger.info("  SMART CCTV ATTENDANCE SYSTEM WITH TAILGATING DETECTION")
        self.logger.info("="*70)
        
        # Initialize all components
        self._initialize_components()
        
        # System state
        self.current_snapshot = 0
        self.last_snapshot_time = time.time()
        self.recognized_students = set()  # Students recognized in current snapshot
        self.session_attendance = {}  # Track attendance across snapshots
        
        # Tailgating detection
        self.tailgating_alert = False
        self.alert_time = 0
        
        self.logger.info("✅ System initialized successfully!")
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Face detection and recognition
        self.logger.info("📥 Loading face detection model...")
        self.face_detector = FaceDetector(
            model_path=self.config['models']['centerface'],
            confidence_threshold=self.config['face_detection']['confidence_threshold']
        )
        
        self.logger.info("📥 Loading face recognition model...")
        self.face_recognizer = FaceRecognizer(
            model_path=self.config['models']['mobilefacenet']
        )
        
        # Person detection and tracking
        self.logger.info("📥 Loading person detection model...")
        self.person_detector = PersonDetector(
            model_path=self.config['models']['ssd_mobilenet'],
            confidence_threshold=self.config['person_detection']['confidence_threshold']
        )
        
        self.logger.info("🎯 Initializing person tracker...")
        self.tracker = CentroidTracker(
            max_disappeared=self.config['tracking']['max_disappeared'],
            max_distance=self.config['tracking']['max_distance']
        )
        
        # Database
        self.logger.info("💾 Loading database...")
        self.database = DatabaseManager(
            students_csv=self.config['database']['students_csv'],
            embeddings_pkl=self.config['database']['embeddings_pkl'],
            attendance_log=self.config['database']['attendance_log']
        )
        
        # FPS counter
        self.fps_counter = FPSCounter()
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame for both attendance and tailgating
        
        Args:
            frame: Input video frame
        
        Returns:
            (annotated_frame, attendance_info, tailgating_info)
        """
        self.fps_counter.update()
        annotated_frame = frame.copy()
        
        # Check if it's time for a new snapshot (every 10 minutes)
        current_time = time.time()
        snapshot_interval = self.config['attendance']['snapshot_interval']
        
        if current_time - self.last_snapshot_time >= snapshot_interval:
            self.current_snapshot += 1
            self.last_snapshot_time = current_time
            self.recognized_students.clear()
            self.logger.info(f"📸 Snapshot {self.current_snapshot} - Starting new attendance check")
        
        # ============ FACE RECOGNITION (Attendance) ============
        faces = self.face_detector.detect(frame)
        recognized_count = 0
        recognized_names = []
        
        for x1, y1, x2, y2, conf in faces:
            # Extract face
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                # Identify student
                student_id, similarity = self.face_recognizer.identify(
                    face_img,
                    self.database.get_all_embeddings(),
                    threshold=self.config['face_recognition']['similarity_threshold']
                )
                
                if student_id:
                    # Get student info
                    student_info = self.database.get_student_info(student_id)
                    name = student_info['name'] if student_info else student_id
                    
                    # Add to recognized set
                    self.recognized_students.add(student_id)
                    recognized_count += 1
                    recognized_names.append(name)
                    
                    # Update session attendance
                    if student_id not in self.session_attendance:
                        self.session_attendance[student_id] = set()
                    self.session_attendance[student_id].add(self.current_snapshot)
                    
                    # Log attendance for this snapshot
                    self.database.log_attendance(
                        student_id=student_id,
                        status='present',
                        confidence=similarity,
                        snapshot_id=self.current_snapshot
                    )
                    
                    # Draw green box (recognized)
                    label = f"{name} ({similarity:.2f})"
                    draw_bbox(annotated_frame, (x1, y1, x2, y2), label, 
                             color=tuple(self.config['display']['bbox_color_recognized']),
                             thickness=self.config['display']['bbox_thickness'])
                else:
                    # Draw red box (unknown)
                    draw_bbox(annotated_frame, (x1, y1, x2, y2), "Unknown", 
                             color=tuple(self.config['display']['bbox_color_unknown']),
                             thickness=self.config['display']['bbox_thickness'])
        
        # ============ PERSON DETECTION (Tailgating) ============
        people = self.person_detector.detect(frame)
        
        # Update tracker
        person_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in people]
        tracked_objects = self.tracker.update(person_boxes)
        
        # Draw tracked people
        for object_id, (x1, y1, x2, y2) in tracked_objects.items():
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated_frame, f"Person {object_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        
        # Check for tailgating
        person_count = len(tracked_objects)
        
        if person_count > recognized_count and recognized_count > 0:
            # TAILGATING DETECTED!
            self.tailgating_alert = True
            self.alert_time = current_time
            
            # Log tailgating event
            self.logger.warning(f"🚨 TAILGATING DETECTED! People: {person_count}, Recognized: {recognized_count}")
            
            # Mark all recognized students as potentially tailgated
            for student_id in self.recognized_students:
                self.database.log_attendance(
                    student_id=student_id,
                    status='present',
                    confidence=1.0,
                    tailgating=True,
                    snapshot_id=self.current_snapshot
                )
        
        # Clear alert after duration
        if self.tailgating_alert and (current_time - self.alert_time) > self.config['tailgating']['alert_duration']:
            self.tailgating_alert = False
        
        # ============ DRAW INFO OVERLAY ============
        self._draw_overlay(annotated_frame, recognized_count, person_count, recognized_names)
        
        # Prepare info dictionaries
        attendance_info = {
            'snapshot': self.current_snapshot,
            'recognized_count': recognized_count,
            'recognized_names': recognized_names,
            'total_enrolled': len(self.database.get_all_students())
        }
        
        tailgating_info = {
            'alert': self.tailgating_alert,
            'person_count': person_count,
            'recognized_count': recognized_count
        }
        
        return annotated_frame, attendance_info, tailgating_info
    
    def _draw_overlay(self, frame: np.ndarray, recognized_count: int, 
                     person_count: int, recognized_names: list):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        fps = self.fps_counter.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Snapshot info
        cv2.putText(frame, f"Snapshot: {self.current_snapshot}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Attendance info
        cv2.putText(frame, f"Recognized: {recognized_count}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Person count
        color = (0, 0, 255) if person_count > recognized_count else (0, 255, 0)
        cv2.putText(frame, f"People Detected: {person_count}", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Tailgating alert
        if self.tailgating_alert:
            alert_text = "⚠️ TAILGATING ALERT! ⚠️"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            
            # Flashing effect
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, alert_text, (text_x, height - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Recognized names (if any)
        if len(recognized_names) > 0 and len(recognized_names) <= 3:
            y_offset = 180
            cv2.putText(frame, "Present:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            for i, name in enumerate(recognized_names[:3]):
                cv2.putText(frame, f"• {name}", (30, y_offset + 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def run(self, camera_source: int = 0):
        """
        Run the system with live camera feed
        
        Args:
            camera_source: Camera index or video file path
        """
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            self.logger.error("❌ Could not open camera!")
            return
        
        self.logger.info("🎥 Starting live feed...")
        self.logger.info("Press 'q' to quit, 's' to save screenshot, 'r' to reset session")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, attendance_info, tailgating_info = self.process_frame(frame)
                
                # Display
                cv2.imshow("Smart CCTV Attendance System", annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    self.logger.info(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reset session
                    self.current_snapshot = 0
                    self.session_attendance.clear()
                    self.recognized_students.clear()
                    self.tracker.reset()
                    self.logger.info("🔄 Session reset!")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final attendance summary
            self._print_attendance_summary()
    
    def _print_attendance_summary(self):
        """Print final attendance summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info("  ATTENDANCE SUMMARY")
        self.logger.info("="*70)
        
        min_snapshots = self.config['attendance']['min_snapshots_for_presence']
        
        for student_id, snapshots in self.session_attendance.items():
            student_info = self.database.get_student_info(student_id)
            name = student_info['name'] if student_info else student_id
            
            status = "PRESENT" if len(snapshots) >= min_snapshots else "ABSENT"
            
            self.logger.info(f"  {name}: {status} ({len(snapshots)} snapshots)")
        
        self.logger.info("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart CCTV Attendance System")
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera source (0 for webcam)')
    
    args = parser.parse_args()
    
    # Run system
    system = SmartCCTVSystem(config_path=args.config)
    system.run(camera_source=args.camera)
