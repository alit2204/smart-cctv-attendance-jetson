import cv2
import os
import glob

def get_available_cameras():
    return sorted(glob.glob("/dev/video*"))

def get_gstreamer_pipeline(device):
    return (f"v4l2src device={device} ! video/x-raw, width=640, height=480 ! videoconvert ! appsink")

# 1. Initialize
available = get_available_cameras()
if not available:
    print("No cameras found!"); exit()

# Select Camera
print("Available:", available)
cam_path = available[0] # Default to first camera

# 2. Setup Person Info
person_name = input("Enter name for this person: ").strip().replace(" ", "_")
folder_path = os.path.join("faces", person_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 3. Open Camera
cap = cv2.VideoCapture(get_gstreamer_pipeline(cam_path), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    cap = cv2.VideoCapture(0) # Fallback

print(f"\n--- CONTROLS ---")
print(f"WINDOW ACTIVE: Press 's' to Save Image")
print(f"WINDOW ACTIVE: Press 'q' to Quit")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clone the frame to draw instructions on it without saving the text to the file
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Images Saved: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press 'S' to Capture | 'Q' to Done", (10, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Dataset Collector", display_frame)

    # Use waitKey to listen for 's' or 'q'
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        img_name = os.path.join(folder_path, f"{person_name}_{count}.jpg")
        cv2.imwrite(img_name, frame) # Save the clean frame (no text)
        print(f"[SAVED] {img_name}")
        count += 1
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Finished. Saved {count} images in {folder_path}")
