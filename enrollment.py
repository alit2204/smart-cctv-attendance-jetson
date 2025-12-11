"""
Student Enrollment Script
Capture student photos and create face embeddings for database
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.append('.')
from src import FaceDetector, FaceRecognizer, DatabaseManager
from src.utils import load_config, setup_logging

class StudentEnrollment:
    """Handle student enrollment process"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize enrollment system"""
        self.config = load_config(config_path)
        setup_logging(
            log_file=self.config['logging']['log_file'],
            level=self.config['logging']['level']
        )
        
        print("="*60)
        print("     SMART CCTV ATTENDANCE - STUDENT ENROLLMENT")
        print("="*60)
        
        # Initialize components
        print("\n📥 Loading models...")
        self.face_detector = FaceDetector(
            model_path=self.config['models']['centerface'],
            confidence_threshold=self.config['face_detection']['confidence_threshold']
        )
        
        self.face_recognizer = FaceRecognizer(
            model_path=self.config['models']['mobilefacenet']
        )
        
        self.database = DatabaseManager(
            students_csv=self.config['database']['students_csv'],
            embeddings_pkl=self.config['database']['embeddings_pkl'],
            attendance_log=self.config['database']['attendance_log']
        )
        
        # Create student photos directory
        self.photos_dir = Path(self.config['student_photos_dir'])
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Enrollment system ready!")
    
    def enroll_student(self, student_id: str, name: str, email: str = "",
                      num_photos: int = 5, camera_source: int = 0):
        """
        Enroll a new student by capturing photos and creating embedding
        
        Args:
            student_id: Unique student ID
            name: Student name
            email: Student email (optional)
            num_photos: Number of photos to capture
            camera_source: Camera index (0 for webcam)
        """
        print(f"\n{'='*60}")
        print(f"   ENROLLING: {name} (ID: {student_id})")
        print(f"{'='*60}")
        
        # Create student directory
        student_dir = self.photos_dir / student_id
        student_dir.mkdir(parents=True, exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print("❌ Error: Could not open camera!")
            return False
        
        print(f"\n📸 Instructions:")
        print(f"   - Position your face in front of the camera")
        print(f"   - Press SPACE to capture photo ({num_photos} photos needed)")
        print(f"   - Turn your head slightly for different angles")
        print(f"   - Press 'q' to cancel\n")
        
        captured_photos = []
        embeddings = []
        photo_count = 0
        
        while photo_count < num_photos:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame!")
                break
            
            # Detect faces
            faces = self.face_detector.detect(frame)
            
            # Draw instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Photos: {photo_count}/{num_photos}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw detected faces
            for x1, y1, x2, y2, conf in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show warning if multiple faces or no face
            if len(faces) == 0:
                cv2.putText(display_frame, "WARNING: No face detected!", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(display_frame, "WARNING: Multiple faces detected!", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Student Enrollment", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture photo on SPACE
            if key == ord(' '):
                if len(faces) == 1:
                    x1, y1, x2, y2, conf = faces[0]
                    
                    # Extract face
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size > 0:
                        # Save photo
                        photo_path = student_dir / f"photo_{photo_count+1}.jpg"
                        cv2.imwrite(str(photo_path), face_img)
                        
                        # Extract embedding
                        embedding = self.face_recognizer.extract_embedding(face_img)
                        embeddings.append(embedding)
                        
                        captured_photos.append(photo_path)
                        photo_count += 1
                        
                        print(f"✅ Captured photo {photo_count}/{num_photos}")
                        
                        # Brief pause
                        cv2.waitKey(200)
                else:
                    print("⚠️  Please ensure only ONE face is visible!")
            
            # Cancel on 'q'
            elif key == ord('q'):
                print("\n❌ Enrollment cancelled!")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate average embedding
        if len(embeddings) > 0:
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-6)
            
            # Add to database
            success = self.database.add_student(
                student_id=student_id,
                name=name,
                email=email,
                face_embedding=avg_embedding
            )
            
            if success:
                print(f"\n✅ Successfully enrolled {name}!")
                print(f"   Student ID: {student_id}")
                print(f"   Photos captured: {len(captured_photos)}")
                print(f"   Photos saved in: {student_dir}")
                return True
            else:
                print(f"\n❌ Failed to enroll student (ID may already exist)")
                return False
        else:
            print("\n❌ No embeddings captured!")
            return False
    
    def batch_enroll_from_csv(self, csv_file: str):
        """
        Enroll multiple students from a CSV file
        
        CSV format: student_id,name,email
        """
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        
        print(f"\n📋 Found {len(df)} students to enroll")
        
        for idx, row in df.iterrows():
            print(f"\n[{idx+1}/{len(df)}] Enrolling {row['name']}...")
            
            success = self.enroll_student(
                student_id=row['student_id'],
                name=row['name'],
                email=row.get('email', '')
            )
            
            if not success:
                choice = input("Continue with next student? (y/n): ")
                if choice.lower() != 'y':
                    break
        
        print(f"\n✅ Batch enrollment completed!")

def main():
    parser = argparse.ArgumentParser(description="Student Enrollment System")
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--id', type=str, help='Student ID')
    parser.add_argument('--name', type=str, help='Student name')
    parser.add_argument('--email', type=str, default='', help='Student email')
    parser.add_argument('--photos', type=int, default=5, 
                       help='Number of photos to capture')
    parser.add_argument('--batch', type=str, 
                       help='CSV file for batch enrollment')
    
    args = parser.parse_args()
    
    # Initialize enrollment system
    enrollment = StudentEnrollment(config_path=args.config)
    
    if args.batch:
        # Batch enrollment
        enrollment.batch_enroll_from_csv(args.batch)
    elif args.id and args.name:
        # Single student enrollment
        enrollment.enroll_student(
            student_id=args.id,
            name=args.name,
            email=args.email,
            num_photos=args.photos
        )
    else:
        # Interactive mode
        print("\n🎓 Interactive Enrollment Mode")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Enroll new student")
            print("2. View enrolled students")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ")
            
            if choice == '1':
                student_id = input("Enter Student ID: ")
                name = input("Enter Name: ")
                email = input("Enter Email (optional): ")
                
                enrollment.enroll_student(
                    student_id=student_id,
                    name=name,
                    email=email
                )
            
            elif choice == '2':
                students = enrollment.database.get_all_students()
                print(f"\n📋 Enrolled Students ({len(students)}):")
                print("-"*60)
                for student in students:
                    print(f"   {student['student_id']}: {student['name']}")
            
            elif choice == '3':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
