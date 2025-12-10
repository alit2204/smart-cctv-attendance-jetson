import cv2
import time

# Open default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    filename = f"image_{counter}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

    counter += 1
    time.sleep(0.3)  # 300 milliseconds

