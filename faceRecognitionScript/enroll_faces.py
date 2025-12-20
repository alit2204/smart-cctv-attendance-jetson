import cv2
import numpy as np
import os
import pickle
import gc

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
RECOGNIZER_PATH = "face_recognition_sface_2021dec.onnx"
FACES_DIR = "faces"
OUTPUT_FILE = "encodings.pickle"

# ==========================================
# 2. INITIALIZE MODELS
# ==========================================
print("[INIT] Loading Hybrid Enrollment Engines...")
# Detection (Caffe - Stable for 4.1.2)
detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
# Recognition (SFace ONNX)
recognizer = cv2.dnn.readNet(RECOGNIZER_PATH)

# Set to CPU Target (Standard for 4.1.2 DNN module)
detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ==========================================
# 3. GPU HELPER FUNCTION
# ==========================================
def gpu_resize(frame, width, height):
    """Offloads resizing to Jetson GPU cores"""
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    gpu_resized = cv2.cuda.resize(gpu_frame, (width, height))
    return gpu_resized.download()

# ==========================================
# 4. PROCESSING LOGIC
# ==========================================
def enroll_faces():
    database = {}
    
    if not os.path.exists(FACES_DIR):
        os.makedirs(FACES_DIR)
        print(f"[!] '{FACES_DIR}' folder created. Add person folders and images first!")
        return

    print(f"[PROCESS] Starting scan of {FACES_DIR} folder...")

    for person_name in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"  -> Processing: {person_name}")
        
        # Take images from the folder
        image_list = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_list:
            print(f"     [SKIP] No images found for {person_name}")
            continue

        for image_name in image_list:
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path)
            if img is None: continue

            h, w = img.shape[:2]
            
            # --- Detection Phase ---
            # Use GPU for initial resize
            resized_300 = gpu_resize(img, 300, 300)
            blob = cv2.dnn.blobFromImage(resized_300, 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            detector.setInput(blob)
            detections = detector.forward()

            found_valid_face = False
            # Find the detection with highest confidence
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.6: # Only accept clear faces
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Crop the face
                    face_roi = img[max(0, startY):endY, max(0, startX):endX]
                    
                    if face_roi.size > 0:
                        # --- Recognition Phase ---
                        # Use GPU for ROI resize
                        face_roi_gpu = gpu_resize(face_roi, 112, 112)
                        rec_blob = cv2.dnn.blobFromImage(face_roi_gpu, 1.0, (112, 112), (0, 0, 0), swapRB=True)
                        
                        recognizer.setInput(rec_blob)
                        features = recognizer.forward()
                        
                        # Store the flattened 128D vector
                        database[person_name] = features.flatten()
                        print(f"     [OK] Enrolled using {image_name}")
                        found_valid_face = True
                        break # Successfully enrolled this person
            
            if found_valid_face:
                break # Move to next person
            else:
                print(f"     [WARN] No face detected in {image_name}")

        # Memory cleanup after each person
        gc.collect()

    # Save to file
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(database, f)
    
    print("-" * 30)
    print(f"[SUCCESS] Database saved to {OUTPUT_FILE}")
    print(f"[TOTAL] {len(database)} people enrolled.")

if __name__ == "__main__":
    enroll_faces()
