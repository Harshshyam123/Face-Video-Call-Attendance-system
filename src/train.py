import cv2
import os
import numpy as np

# ----------------------------
# PATH SETUP (IMPORTANT)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "..", "dataset")

print("TRAINING STARTED")

faces = []
labels = []
label_map = {}
current_label = 0

# ----------------------------
# DATASET READ
# ----------------------------
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# ----------------------------
# SAVE MODEL & LABELS
# ----------------------------
model.save(os.path.join(BASE_DIR, "face_model.yml"))
np.save(os.path.join(BASE_DIR, "labels.npy"), label_map)

print("TRAINING COMPLETE ✅")
