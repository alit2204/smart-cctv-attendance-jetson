"""
Gradio Web Dashboard for Smart CCTV Attendance System
Beautiful web interface for monitoring attendance and tailgating
"""

import gradio as gr
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.append('.')
from src import (FaceDetector, FaceRecognizer, PersonDetector,
                CentroidTracker, DatabaseManager)
from src.utils import load_config, setup_logging

# Global system state
class SystemState:
    def __init__(self):
        self.config = load_config()
        setup_logging(level=self.config['logging']['level'])
        
        # Initialize components
        self.face_detector = FaceDetector(
            model_path=self.config['models']['centerface'],
            confidence_threshold=0.5
        )
        
        self.face_recognizer = FaceRecognizer(
            model_path=self.config['models']['mobilefacenet']
        )
        
        self.person_detector = PersonDetector(
            model_path=self.config['models']['ssd_mobilenet'],
            confidence_threshold=0.4
        )
        
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=50)
        
        self.database = DatabaseManager(
            students_csv=self.config['database']['students_csv'],
            embeddings_pkl=self.config['database']['embeddings_pkl'],
            attendance_log=self.config['database']['attendance_log']
        )
        
        self.current_snapshot = 0
        self.last_snapshot_time = time.time()
        self.session_attendance = {}
        self.tailgating_alerts = []

state = SystemState()

def process_frame_for_gradio(frame):
    """Process frame and return annotated image with info"""
    if frame is None:
        return None, "No frame", "0", "0", "No alerts"
    
    # Check for snapshot interval
    current_time = time.time()
    if current_time - state.last_snapshot_time >= 600:  # 10 minutes
        state.current_snapshot += 1
        state.last_snapshot_time = current_time
    
    # Detect faces
    faces = state.face_detector.detect(frame)
    recognized_count = 0
    recognized_names = []
    
    for x1, y1, x2, y2, conf in faces:
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size > 0:
            student_id, similarity = state.face_recognizer.identify(
                face_img,
                state.database.get_all_embeddings(),
                threshold=0.6
            )
            
            if student_id:
                student_info = state.database.get_student_info(student_id)
                name = student_info['name'] if student_info else student_id
                
                recognized_count += 1
                recognized_names.append(name)
                
                # Update session
                if student_id not in state.session_attendance:
                    state.session_attendance[student_id] = set()
                state.session_attendance[student_id].add(state.current_snapshot)
                
                # Draw green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Detect people
    people = state.person_detector.detect(frame)
    person_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in people]
    tracked_objects = state.tracker.update(person_boxes)
    
    for object_id, (x1, y1, x2, y2) in tracked_objects.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, f"P{object_id}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    
    person_count = len(tracked_objects)
    
    # Check tailgating
    alert_text = "✅ No Tailgating"
    if person_count > recognized_count and recognized_count > 0:
        alert_text = f"🚨 TAILGATING! {person_count-recognized_count} unauthorized"
        state.tailgating_alerts.append({
            'time': time.strftime('%H:%M:%S'),
            'people': person_count,
            'recognized': recognized_count
        })
    
    # Draw overlay
    cv2.putText(frame, f"Snapshot: {state.current_snapshot}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Recognized: {recognized_count}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Total People: {person_count}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Info for display
    names_text = ", ".join(recognized_names) if recognized_names else "None"
    
    return frame, names_text, str(recognized_count), str(person_count), alert_text

def get_attendance_table():
    """Get current attendance as DataFrame"""
    data = []
    
    for student_id, snapshots in state.session_attendance.items():
        student_info = state.database.get_student_info(student_id)
        name = student_info['name'] if student_info else student_id
        
        status = "Present" if len(snapshots) >= 4 else "Partial"
        
        data.append({
            'ID': student_id,
            'Name': name,
            'Snapshots': len(snapshots),
            'Status': status
        })
    
    return pd.DataFrame(data) if data else pd.DataFrame(columns=['ID', 'Name', 'Snapshots', 'Status'])

def get_alerts_table():
    """Get tailgating alerts as DataFrame"""
    return pd.DataFrame(state.tailgating_alerts[-10:]) if state.tailgating_alerts else pd.DataFrame(columns=['time', 'people', 'recognized'])

def reset_session():
    """Reset the attendance session"""
    state.current_snapshot = 0
    state.session_attendance.clear()
    state.tailgating_alerts.clear()
    state.tracker.reset()
    return "✅ Session reset successfully!"

# Create Gradio interface
with gr.Blocks(title="Smart CCTV Attendance System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 Smart CCTV Attendance System with Tailgating Detection
    ### Real-time face recognition attendance + security monitoring
    """)
    
    with gr.Tab("📹 Live Monitoring"):
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Image(source="webcam", streaming=True, label="Live Feed")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Current Status")
                recognized_names = gr.Textbox(label="Recognized Students", value="None")
                recognized_count_display = gr.Textbox(label="Recognized Count", value="0")
                person_count_display = gr.Textbox(label="Total People", value="0")
                alert_display = gr.Textbox(label="Security Alert", value="No alerts")
                
                gr.Markdown("### 📋 Session Attendance")
                attendance_table = gr.Dataframe(
                    headers=['ID', 'Name', 'Snapshots', 'Status'],
                    label="Current Session"
                )
                
                refresh_btn = gr.Button("🔄 Refresh Tables")
        
        # Update video processing
        video_input.stream(
            fn=process_frame_for_gradio,
            inputs=[video_input],
            outputs=[video_input, recognized_names, recognized_count_display, 
                    person_count_display, alert_display],
            show_progress=False
        )
        
        # Refresh tables
        refresh_btn.click(
            fn=get_attendance_table,
            outputs=[attendance_table]
        )
    
    with gr.Tab("📊 Attendance Records"):
        gr.Markdown("### Today's Attendance Summary")
        
        with gr.Row():
            date_input = gr.Textbox(label="Date (YYYY-MM-DD)", 
                                   value=time.strftime('%Y-%m-%d'))
            load_btn = gr.Button("📥 Load Records")
        
        attendance_summary = gr.Dataframe(label="Attendance Log")
        
        def load_attendance(date):
            summary = state.database.get_attendance_summary(date)
            return summary
        
        load_btn.click(
            fn=load_attendance,
            inputs=[date_input],
            outputs=[attendance_summary]
        )
    
    with gr.Tab("🚨 Security Alerts"):
        gr.Markdown("### Tailgating Detection Log")
        
        alerts_table = gr.Dataframe(
            headers=['time', 'people', 'recognized'],
            label="Recent Alerts"
        )
        
        refresh_alerts_btn = gr.Button("🔄 Refresh Alerts")
        
        refresh_alerts_btn.click(
            fn=get_alerts_table,
            outputs=[alerts_table]
        )
    
    with gr.Tab("👥 Student Database"):
        gr.Markdown("### Enrolled Students")
        
        students_table = gr.Dataframe(label="Students")
        
        load_students_btn = gr.Button("📥 Load Students")
        
        def load_students():
            students = state.database.get_all_students()
            return pd.DataFrame(students)
        
        load_students_btn.click(
            fn=load_students,
            outputs=[students_table]
        )
    
    with gr.Tab("⚙️ System Control"):
        gr.Markdown("### Session Management")
        
        reset_btn = gr.Button("🔄 Reset Session", variant="primary")
        reset_status = gr.Textbox(label="Status")
        
        reset_btn.click(
            fn=reset_session,
            outputs=[reset_status]
        )
        
        gr.Markdown("""
        ### 📝 Instructions:
        - **Live Monitoring**: View real-time detection and attendance
        - **Attendance Records**: View historical attendance logs
        - **Security Alerts**: Monitor tailgating detection events
        - **Student Database**: Manage enrolled students
        - **Reset Session**: Clear current session data
        
        ### 🎯 Snapshot System:
        - Attendance is captured every 10 minutes (snapshot)
        - Students must appear in ≥4 snapshots to be marked Present
        - Tailgating alerts trigger when people count > recognized faces
        """)

if __name__ == "__main__":
    print("="*70)
    print("  🚀 Starting Smart CCTV Attendance System Dashboard")
    print("="*70)
    print("\n📱 Opening web interface...")
    print("   The dashboard will open in your browser automatically")
    print("   URL: http://localhost:7860")
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set True to create public link
    )
