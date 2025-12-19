import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from glob import glob

# -----------------------
# CONFIG
# -----------------------
ENGINE_PATH = "MobileNet-v2_fp16.trt"
FACE_DATASET_PATH = "faces"   # dataset path
CAMERA_ID = 0                  # USB camera ID
INPUT_SHAPE = (224, 224)       # depending on your MobileNet-v2 variant
THRESHOLD = 0.6                # similarity threshold for recognition

# -----------------------
# HELPER FUNCTIONS
# -----------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, INPUT_SHAPE)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0  # normalize
    face_img = np.transpose(face_img, (2, 0, 1))     # HWC -> CHW
    return face_img

def infer(context, inputs, outputs, bindings, stream, image):
    np.copyto(inputs[0][0], image.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
    return outputs[0][0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_face_database():
    db = {}
    for person_dir in os.listdir(FACE_DATASET_PATH):
        person_path = os.path.join(FACE_DATASET_PATH, person_dir)
        img_files = glob(os.path.join(person_path, "*.jpg"))
        embeddings = []
        for img_file in img_files:
            img = cv2.imread(img_file)
            img_proc = preprocess_face(img)
            embedding = infer(context, inputs, outputs, bindings, stream, img_proc)
            embeddings.append(embedding)
        db[person_dir] = np.mean(embeddings, axis=0)
    return db

# -----------------------
# LOAD ENGINE AND DATABASE
# -----------------------
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)
face_db = load_face_database()
print("Face database loaded:", list(face_db.keys()))

# -----------------------
# FACE DETECTION (OpenCV DNN)
# -----------------------
detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # you can download lightweight OpenCV face detector
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# -----------------------
# REAL-TIME CAMERA LOOP
# -----------------------
cap = cv2.VideoCapture(CAMERA_ID)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    # Prepare input blob for face detector
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_input = preprocess_face(face)
            emb = infer(context, inputs, outputs, bindings, stream, face_input)

            # Compare with database
            name = "Unknown"
            best_score = 0
            for person, db_emb in face_db.items():
                score = cosine_similarity(emb, db_emb)
                if score > best_score:
                    best_score = score
                    if best_score > THRESHOLD:
                        name = person

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

