"""
Database Manager
Handles student database, face embeddings, and attendance logging
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages student database, face embeddings, and attendance records
    """
    
    def __init__(self, students_csv: str = "database/students.csv",
                 embeddings_pkl: str = "database/embeddings.pkl",
                 attendance_log: str = "database/attendance_log.csv"):
        """
        Initialize Database Manager
        
        Args:
            students_csv: Path to students CSV file
            embeddings_pkl: Path to face embeddings pickle file
            attendance_log: Path to attendance log CSV
        """
        self.students_csv = students_csv
        self.embeddings_pkl = embeddings_pkl
        self.attendance_log = attendance_log
        
        # Create database directory
        Path(students_csv).parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create databases
        self.students_df = self._load_students()
        self.embeddings = self._load_embeddings()
        
        logger.info(f"✅ Database Manager initialized")
        logger.info(f"   Students: {len(self.students_df)}")
        logger.info(f"   Embeddings: {len(self.embeddings)}")
    
    def _load_students(self) -> pd.DataFrame:
        """Load students database"""
        if Path(self.students_csv).exists():
            df = pd.read_csv(self.students_csv)
            logger.info(f"Loaded {len(df)} students from {self.students_csv}")
        else:
            # Create empty database
            df = pd.DataFrame(columns=['student_id', 'name', 'email', 'enrollment_date'])
            logger.info(f"Created new students database at {self.students_csv}")
        
        return df
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load face embeddings"""
        if Path(self.embeddings_pkl).exists():
            with open(self.embeddings_pkl, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded {len(embeddings)} embeddings from {self.embeddings_pkl}")
        else:
            embeddings = {}
            logger.info(f"Created new embeddings database")
        
        return embeddings
    
    def add_student(self, student_id: str, name: str, email: str = "",
                   face_embedding: Optional[np.ndarray] = None):
        """
        Add a new student to database
        
        Args:
            student_id: Unique student ID
            name: Student name
            email: Student email (optional)
            face_embedding: 128-D face embedding (optional)
        """
        # Check if student already exists
        if student_id in self.students_df['student_id'].values:
            logger.warning(f"Student {student_id} already exists!")
            return False
        
        # Add to students dataframe
        new_student = {
            'student_id': student_id,
            'name': name,
            'email': email,
            'enrollment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.students_df = pd.concat([self.students_df, pd.DataFrame([new_student])], 
                                    ignore_index=True)
        
        # Add face embedding if provided
        if face_embedding is not None:
            self.embeddings[student_id] = face_embedding
        
        # Save to disk
        self.save()
        
        logger.info(f"✅ Added student: {student_id} - {name}")
        return True
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information by ID"""
        student = self.students_df[self.students_df['student_id'] == student_id]
        
        if len(student) == 0:
            return None
        
        return student.iloc[0].to_dict()
    
    def get_all_students(self) -> List[Dict]:
        """Get all students"""
        return self.students_df.to_dict('records')
    
    def update_embedding(self, student_id: str, face_embedding: np.ndarray):
        """
        Update face embedding for a student
        
        Args:
            student_id: Student ID
            face_embedding: New 128-D face embedding
        """
        if student_id not in self.students_df['student_id'].values:
            logger.error(f"Student {student_id} not found!")
            return False
        
        self.embeddings[student_id] = face_embedding
        self.save_embeddings()
        
        logger.info(f"✅ Updated embedding for student {student_id}")
        return True
    
    def get_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """Get face embedding for a student"""
        return self.embeddings.get(student_id)
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all face embeddings"""
        return self.embeddings
    
    def log_attendance(self, student_id: str, status: str = "present",
                      confidence: float = 0.0, tailgating: bool = False,
                      snapshot_id: int = 0):
        """
        Log attendance record
        
        Args:
            student_id: Student ID
            status: 'present', 'absent', 'late'
            confidence: Recognition confidence score
            tailgating: Whether tailgating was detected
            snapshot_id: Snapshot number (for AttenFace logic)
        """
        # Create attendance record
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'student_id': student_id,
            'status': status,
            'confidence': confidence,
            'tailgating': tailgating,
            'snapshot_id': snapshot_id
        }
        
        # Append to log file
        df_record = pd.DataFrame([record])
        
        if Path(self.attendance_log).exists():
            df_record.to_csv(self.attendance_log, mode='a', header=False, index=False)
        else:
            df_record.to_csv(self.attendance_log, mode='w', header=True, index=False)
        
        logger.debug(f"Logged attendance: {student_id} - {status}")
    
    def get_attendance_summary(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get attendance summary for a date
        
        Args:
            date: Date in 'YYYY-MM-DD' format (None for today)
        
        Returns:
            DataFrame with attendance summary
        """
        if not Path(self.attendance_log).exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.attendance_log)
        
        # Filter by date
        if date:
            df = df[df['timestamp'].str.startswith(date)]
        else:
            today = datetime.now().strftime('%Y-%m-%d')
            df = df[df['timestamp'].str.startswith(today)]
        
        return df
    
    def calculate_class_attendance(self, class_duration_minutes: int = 60,
                                  snapshot_interval_minutes: int = 10,
                                  min_snapshots: int = 4) -> Dict[str, str]:
        """
        Calculate final attendance based on snapshot logic (AttenFace style)
        
        Args:
            class_duration_minutes: Total class duration
            snapshot_interval_minutes: Interval between snapshots
            min_snapshots: Minimum snapshots to be marked present
        
        Returns:
            Dictionary of {student_id: 'present' or 'absent'}
        """
        if not Path(self.attendance_log).exists():
            return {}
        
        # Get today's attendance
        df = self.get_attendance_summary()
        
        if len(df) == 0:
            return {}
        
        # Count snapshots per student
        snapshot_counts = df.groupby('student_id')['snapshot_id'].nunique()
        
        # Determine final status
        final_attendance = {}
        for student_id, count in snapshot_counts.items():
            if count >= min_snapshots:
                final_attendance[student_id] = 'present'
            else:
                final_attendance[student_id] = 'absent'
        
        return final_attendance
    
    def save(self):
        """Save all databases to disk"""
        self.save_students()
        self.save_embeddings()
    
    def save_students(self):
        """Save students database"""
        self.students_df.to_csv(self.students_csv, index=False)
        logger.debug(f"Saved students database to {self.students_csv}")
    
    def save_embeddings(self):
        """Save face embeddings"""
        with open(self.embeddings_pkl, 'wb') as f:
            pickle.dump(self.embeddings, f)
        logger.debug(f"Saved embeddings to {self.embeddings_pkl}")
    
    def export_attendance_report(self, output_file: str = "attendance_report.csv",
                                date: Optional[str] = None):
        """
        Export attendance report with student names
        
        Args:
            output_file: Output CSV file path
            date: Date to export (None for today)
        """
        df = self.get_attendance_summary(date)
        
        if len(df) == 0:
            logger.warning("No attendance records found")
            return
        
        # Merge with student info
        merged = df.merge(self.students_df[['student_id', 'name']], 
                         on='student_id', how='left')
        
        # Save report
        merged.to_csv(output_file, index=False)
        logger.info(f"✅ Exported attendance report to {output_file}")

if __name__ == "__main__":
    # Test database manager
    import sys
    sys.path.append('..')
    from src.utils import setup_logging
    
    setup_logging(level="INFO")
    
    # Initialize database
    db = DatabaseManager()
    
    # Add test students
    print("\n1️⃣ Adding test students...")
    db.add_student("S001", "Alice Johnson", "alice@example.com")
    db.add_student("S002", "Bob Smith", "bob@example.com")
    db.add_student("S003", "Charlie Brown", "charlie@example.com")
    
    # Add test embeddings
    print("\n2️⃣ Adding test embeddings...")
    test_embedding = np.random.rand(128)
    db.update_embedding("S001", test_embedding)
    db.update_embedding("S002", test_embedding)
    
    # Log attendance
    print("\n3️⃣ Logging attendance...")
    db.log_attendance("S001", "present", confidence=0.95, snapshot_id=1)
    db.log_attendance("S001", "present", confidence=0.92, snapshot_id=2)
    db.log_attendance("S002", "present", confidence=0.88, snapshot_id=1)
    
    # Get summary
    print("\n4️⃣ Attendance Summary:")
    summary = db.get_attendance_summary()
    print(summary)
    
    # Calculate final attendance
    print("\n5️⃣ Final Attendance:")
    final = db.calculate_class_attendance(min_snapshots=2)
    for student_id, status in final.items():
        info = db.get_student_info(student_id)
        print(f"   {info['name']}: {status}")
    
    print("\n✅ Database manager test completed!")
