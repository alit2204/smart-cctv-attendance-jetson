import cv2
import numpy as np
import os

labels = {}
faces_dir = "faces"  # folder with subfolders
for idx, person in enumerate(os.listdir(faces_dir)):
    labels[idx] = person


# Haarcascade path for JetPack 4.1
face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

# Load trained LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")


# Use GStreamer camera pipeline (fixes Jetson Nano camera errors)
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

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

