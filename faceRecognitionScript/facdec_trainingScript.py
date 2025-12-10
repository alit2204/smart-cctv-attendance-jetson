import cv2
import os
import numpy as np

DATASET_DIR = "faces"

recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

images = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_dir):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces = face_cascade.detectMultiScale(img, 1.2, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            images.append(face)
            labels.append(current_label)

    current_label += 1

recognizer.train(images, np.array(labels))
recognizer.save("face_model.yml")

# Save labels
with open("labels.txt", "w") as f:
    for id, name in label_map.items():
        f.write(f"{id},{name}\n")

print("[INFO] Training complete.")

