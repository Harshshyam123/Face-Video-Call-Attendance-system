import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

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

attendance = {}
camera_running = False

# ----------------------------
# CAMERA FUNCTION (BACKEND)
# ----------------------------
def start_camera():
    global camera_running
    camera_running = True

    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            label, confidence = model.predict(roi)
            name = "Unknown"

            if confidence < 80:
                name = labels[label]
                if name not in attendance:
                    attendance[name] = datetime.now().strftime("%H:%M:%S")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Video Call Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# STOP CAMERA
# ----------------------------
def stop_camera():
    global camera_running
    camera_running = False
    messagebox.showinfo("Info", "Camera stopped")

# ----------------------------
# SAVE ATTENDANCE
# ----------------------------
def save_attendance():
    if not attendance:
        messagebox.showwarning("Warning", "No attendance to save")
        return

    df = pd.DataFrame(list(attendance.items()), columns=["Name", "Time"])
    df["Date"] = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(attendance_path, index=False)

    messagebox.showinfo("Success", "Attendance saved successfully")

# ----------------------------
# UI DESIGN (FRONTEND)
# ----------------------------
root = tk.Tk()
root.title("Video Call Attendance System")
root.geometry("400x300")

title = tk.Label(root, text="Attendance System", font=("Arial", 18, "bold"))
title.pack(pady=20)

btn_start = tk.Button(root, text="Start Camera", width=20, height=2, command=start_camera)
btn_start.pack(pady=10)

btn_stop = tk.Button(root, text="Stop Camera", width=20, height=2, command=stop_camera)
btn_stop.pack(pady=10)

btn_save = tk.Button(root, text="Save Attendance", width=20, height=2, command=save_attendance)
btn_save.pack(pady=10)

root.mainloop()
