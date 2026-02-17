import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ----------------------------
# PATH SETUP
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "face_model.yml")
labels_path = os.path.join(BASE_DIR, "labels.npy")
attendance_path = os.path.join(BASE_DIR, "..", "attendance", "attendance.csv")

# ----------------------------
# LOAD MODEL & LABELS
# ----------------------------
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

labels = np.load(labels_path, allow_pickle=True).item()

# ----------------------------
# FACE DETECTOR
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# CAMERA START
# ----------------------------
cap = cv2.VideoCapture(0)
attendance = {}

print("CAMERA STARTED")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

        label, confidence = model.predict(roi)

        name = "Unknown"

        if confidence < 80:
            name = labels[label]

            if name not in attendance:
                attendance[name] = datetime.now().strftime("%H:%M:%S")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Video Call Attendance System", frame)

    # PRESS q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# CAMERA STOP
# ----------------------------
cap.release()
cv2.destroyAllWindows()

# ----------------------------
# SAVE ATTENDANCE
# ----------------------------
if attendance:
    df = pd.DataFrame(list(attendance.items()), columns=["Name", "Time"])
    df["Date"] = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(attendance_path, index=False)

print("ATTENDANCE SAVED ✅")
