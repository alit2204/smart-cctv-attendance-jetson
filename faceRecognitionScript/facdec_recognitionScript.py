import cv2
import numpy as np
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# -------------------------------------------------
# Google Sheet Setup (using your provided JSON file)
# -------------------------------------------------

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "facedetectionattendece-e013eda423c6_googleSheetAccesskey.json",
    scope
)

client = gspread.authorize(creds)

# Your actual Google Sheet ID
sheet = client.open_by_key("1UGBl5Ic-Dqeo8dRGaTycK237Yh9eEVAGMajreZKoSZk").sheet1


def mark_attendance(name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Row format: [A, B, C]
    sheet.append_row(["", name, timestamp])
    print(f"[LOG] Attendance marked for {name} at {timestamp}")



# -------------------------------------------------
# Face Recognition Setup
# -------------------------------------------------

labels = {}
faces_dir = "faces"

for idx, person in enumerate(os.listdir(faces_dir)):
    labels[idx] = person

face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")


def gstreamer_pipeline():
    return (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! appsink"
    )


cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[ERROR] Cannot open USB camera")
    exit()

print("[INFO] Starting real-time face recognition...")

attendance_recorded = set()

# -------------------------------------------------
# Main loop
# -------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Can't read frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(face_roi)

        if confidence < 80:
            name = labels.get(id_, "unknown")
        else:
            name = "unknown"

        if name != "unknown" and name not in attendance_recorded:
            mark_attendance(name)
            attendance_recorded.add(name)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

