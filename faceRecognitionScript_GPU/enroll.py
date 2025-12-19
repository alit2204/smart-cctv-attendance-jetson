import cv2
import numpy as np
from trt_engine import TRTModule

arcface = TRTModule("arcface_fp16.trt")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

embeddings, names = [], []
name = input("Enter person name: ")

while len(embeddings) < 15:
    ret, frame = cap.read()
    face = cv2.resize(frame, (112,112))
    face = face[:,:,::-1]
    face = face.transpose(2,0,1)[None].astype(np.float32)
    face = (face - 127.5) / 128.0

    emb = arcface.infer(face)
    emb = emb / np.linalg.norm(emb)

    embeddings.append(emb)
    names.append(name)

    cv2.imshow("Enroll", frame)
    if cv2.waitKey(1) == 27:
        break

np.savez("face_db.npz",
         embeddings=np.array(embeddings),
         names=np.array(names))

cap.release()
cv2.destroyAllWindows()

