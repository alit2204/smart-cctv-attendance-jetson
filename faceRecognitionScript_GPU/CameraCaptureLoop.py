import cv2
import os
import time

# --------------------------
DATASET_DIR = "dataset"
INTERVAL = 0.5
IMG_WIDTH = 640
IMG_HEIGHT = 480

name = input("Enter the name of the person: ").strip()
if not name:
    print("Name cannot be empty!")
    exit(1)

person_dir = os.path.join(DATASET_DIR, name)
os.makedirs(person_dir, exist_ok=True)

print(f"[INFO] Capturing images for {name} into '{person_dir}'")
print("Press ESC to stop capturing early.")

# --------------------------
# Use USB webcam (0 = first camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("[ERROR] Could not open USB webcam. Check connection.")

img_count = 0

# --------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        cv2.putText(frame, f"Frame: {img_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Camera Capture", frame)

        img_name = f"{img_count:04d}.jpg"
        cv2.imwrite(os.path.join(person_dir, img_name), frame)
        img_count += 1

        key = cv2.waitKey(int(INTERVAL * 1000)) & 0xFF
        if key == 27:  # ESC key
            print("[INFO] Capture stopped by user")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Total images captured: {img_count}")

