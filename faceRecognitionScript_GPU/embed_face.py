import os
import numpy as np
import cv2
from trt_engine import TRTModule

# ------------------------
# CONFIG
# ------------------------
FACES_DIR = "./faces"           # faces/<name>/img.jpg
DB_FILE = "face_db.npz"
ARC_TRT = "arcface_fp16.trt"
IMAGE_SIZE = (112, 112)         # ArcFace input size

# ------------------------
# LOAD ARCFACE
# ------------------------
arcface = TRTModule(ARC_TRT)

# ------------------------
# LOAD EXISTING DATABASE
# ------------------------
if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(db["embeddings"])
    names = list(db["names"])
    processed_files = set(db.get("files", []))
else:
    embeddings = []
    names = []
    processed_files = set()

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def preprocess_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    face = cv2.resize(img, IMAGE_SIZE)
    face = face[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    face = (face - 127.5) / 128.0
    return face

def get_all_images(folder):
    """Return list of (name, filepath) tuples for all images"""
    img_list = []
    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)
        if not os.path.isdir(person_path):
            continue
        for fname in os.listdir(person_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_list.append((person_name, os.path.join(person_path, fname)))
    return img_list

# ------------------------
# PROCESS NEW IMAGES
# ------------------------
all_images = get_all_images(FACES_DIR)
new_images = [(n,p) for n,p in all_images if p not in processed_files]

if not new_images:
    print("No new images to process.")
else:
    print(f"Processing {len(new_images)} new images...")

for name, img_path in new_images:
    face = preprocess_face(img_path)
    if face is None:
        print(f"Skipping {img_path}: cannot read image.")
        continue

    emb = arcface.infer(face)
    emb = emb / np.linalg.norm(emb)

    embeddings.append(emb)
    names.append(name)
    processed_files.add(img_path)

# ------------------------
# SAVE DATABASE
# ------------------------
np.savez(DB_FILE,
         embeddings=np.array(embeddings),
         names=np.array(names),
         files=np.array(list(processed_files))
         )

print(f"Database updated. Total embeddings: {len(names)}")

