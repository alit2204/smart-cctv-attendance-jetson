import os
import sys

# --- 0. SILENCE DRIVER WARNINGS ---
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import threading
import gc
import time

# --- 1. INITIAL USER PROMPTS ---
print("--- Wide-Angle HD Security & Attendance System ---")
enable_tailgating = input("Enable security detection? (y/n): ").lower().strip() == 'y'
thresh_input = input("Enter recognition threshold % (default 60): ").strip()
REC_THRESHOLD = float(thresh_input) / 100.0 if thresh_input.isdigit() else 0.60

# --- 2. SETTINGS & PATHS ---
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
RECOGNIZER_PATH = "face_recognition_sface_2021dec.onnx"
SHEET_KEY = "1UGBl5Ic-Dqeo8dRGaTycK237Yh9eEVAGMajreZKoSZk"
CREDS_FILE = "facedetectionattendece-c42612273f13.json"

CROSS_LINE_Y = 0.75  # Tripwire at 75% height
GRACE_PERIOD = 5     # Seconds the "gate" stays open after recognition
authorized_entry_timer = 0
last_security_log = datetime.min

# --- 3. INITIALIZE MODELS ---
os.system("sudo nvpmodel -m 0")
os.system("sudo jetson_clocks")

detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
recognizer = cv2.dnn.readNet(RECOGNIZER_PATH)

with open("encodings.pickle", "rb") as f:
    database = pickle.load(f)

# Google Sheets Setup
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
sheet = None

def connect_sheets():
    global sheet
    try:
        sheet = gspread.authorize(creds).open_by_key(SHEET_KEY).sheet1
        print("[SUCCESS] Google Sheets connected.")
    except Exception as e:
        print(f"[WARN] Sheets connection failed: {e}")

threading.Thread(target=connect_sheets, daemon=True).start()
attendance_cooldown = {}

# --- 4. HELPERS ---
def gpu_resize(frame, width, height):
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    gpu_resized = cv2.cuda.resize(gpu_frame, (width, height), interpolation=cv2.INTER_CUBIC)
    return gpu_resized.download()

# --- 5. CAMERA INITIALIZATION (FULL HD) ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("[CRITICAL] Camera Hardware not found.")
    sys.exit()

win_name = "Attendance System"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# --- 6. MAIN LOOP ---
prev_frame_time = 0

while True:
    new_frame_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    inf_start = time.time()
    
    # "Gate" Logic: Open if an authorized person was recently seen
    is_open = (time.time() - authorized_entry_timer < GRACE_PERIOD)
    line_y = int(CROSS_LINE_Y * h)
    
    if enable_tailgating:
        line_color = (0, 255, 0) if is_open else (0, 0, 255)
        cv2.line(frame, (0, line_y), (w, line_y), line_color, 1)

    # Face Detection
    blob = cv2.dnn.blobFromImage(gpu_resize(frame, 300, 300), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    security_alert_active = False

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.45:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            roi = frame[startY:endY, startX:endX]
            if roi.size > 0 and roi.shape[0] > 20:
                rec_blob = cv2.dnn.blobFromImage(gpu_resize(roi, 112, 112), 1.0, (112, 112), (0, 0, 0), swapRB=True)
                recognizer.setInput(rec_blob)
                features = recognizer.forward().flatten()

                best_name, max_score = "Unknown", 0.0
                for name, saved_feat in database.items():
                    score = np.dot(features, saved_feat) / (np.linalg.norm(features) * np.linalg.norm(saved_feat))
                    if score > max_score:
                        max_score, best_name = score, name

                score_pct = int(max_score * 100)
                is_below_thresh = max_score < REC_THRESHOLD
                is_crossing = ((startY + endY) / 2 > line_y)

                # --- DECISION LOGIC ---
                
                # A: AUTHORIZED PERSON
                if not is_below_thresh:
                    color = (0, 255, 0)
                    label = f"{best_name} ({score_pct}%)"
                    authorized_entry_timer = time.time() # Open the "Gate"
                    
                    now = datetime.now()
                    if best_name not in attendance_cooldown or (now - attendance_cooldown[best_name]) > timedelta(minutes=5):
                        attendance_cooldown[best_name] = now
                        if sheet:
                            ts = now.strftime("%Y-%m-%d %H:%M:%S")
                            threading.Thread(target=lambda: sheet.append_row(["AUTHORIZED", best_name, ts])).start()

                # B: TAILGATER (Unknown crossing while gate is OPEN)
                elif is_below_thresh and is_crossing and is_open:
                    color = (0, 0, 255)
                    label = "TAILGATER"
                    security_alert_active = True
                    now = datetime.now()
                    if (now - last_security_log) > timedelta(seconds=10):
                        last_security_log = now
                        if sheet:
                            ts = now.strftime("%Y-%m-%d %H:%M:%S")
                            threading.Thread(target=lambda: sheet.append_row(["SECURITY", "TAILGATE (PIGGYBACKING)", ts])).start()

                # C: UNAUTHORIZED ENTRY (Unknown crossing while gate is CLOSED)
                elif is_below_thresh and is_crossing and not is_open:
                    color = (0, 0, 255)
                    label = "UNAUTHORIZED ENTRY"
                    security_alert_active = True
                    now = datetime.now()
                    if (now - last_security_log) > timedelta(seconds=10):
                        last_security_log = now
                        if sheet:
                            ts = now.strftime("%Y-%m-%d %H:%M:%S")
                            threading.Thread(target=lambda: sheet.append_row(["SECURITY", "UNAUTHORIZED ENTRY", ts])).start()

                # D: UNKNOWN (Not crossing line)
                else:
                    color = (0, 165, 255) if max_score >= 0.35 else (0, 0, 255)
                    label = f"Unknown ({score_pct}%)"

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
                cv2.putText(frame, label, (startX, startY-7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Visual Alert Overlay
    if security_alert_active:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 10)
        cv2.putText(frame, "!!! SECURITY BREACH !!!", (w//2-180, h//2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

    # --- HUD & INSTRUCTIONS ---
    inf_time = (time.time() - inf_start) * 1000
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    cv2.rectangle(frame, (0, 0), (w, 65), (30, 30, 30), -1) 
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Latency: {inf_time:.1f}ms", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    gate_status = "GATE: OPEN" if is_open else "GATE: LOCKED"
    cv2.putText(frame, gate_status, (w//2-80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100) if is_open else (100, 100, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, "GUIDE: Face Camera Directly", (w-220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Press 'Q' to shutdown", (w-220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    gc.collect()

cap.release()
cv2.destroyAllWindows()
