import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_model.yml")
labels_path = os.path.join(BASE_DIR, "labels.npy")
attendance_path = os.path.join(BASE_DIR, "..", "attendance", "attendance.csv")

# ---------------- LOAD MODEL ----------------
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)
labels = np.load(labels_path, allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

attendance = {}
camera_on = False

# ---------------- BUTTON AREAS ----------------
buttons = {
    "start": (20, 20, 180, 70),
    "stop": (200, 20, 360, 70),
    "save": (380, 20, 540, 70),
    "exit": (560, 20, 720, 70)
}

def draw_buttons(frame):
    cv2.rectangle(frame, (20,20), (180,70), (0,255,0), -1)
    cv2.rectangle(frame, (200,20), (360,70), (0,0,255), -1)
    cv2.rectangle(frame, (380,20), (540,70), (255,0,0), -1)
    cv2.rectangle(frame, (560,20), (720,70), (50,50,50), -1)

    cv2.putText(frame,"START",(45,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
    cv2.putText(frame,"STOP",(235,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame,"SAVE",(415,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(frame,"EXIT",(600,55),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

def inside(x, y, box):
    x1,y1,x2,y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def mouse_click(event, x, y, flags, param):
    global camera_on
    if event == cv2.EVENT_LBUTTONDOWN:
        if inside(x,y,buttons["start"]):
            camera_on = True
        elif inside(x,y,buttons["stop"]):
            camera_on = False
        elif inside(x,y,buttons["save"]):
            if attendance:
                df = pd.DataFrame(list(attendance.items()), columns=["Name","Time"])
                df["Date"] = datetime.now().strftime("%Y-%m-%d")
                df.to_csv(attendance_path, index=False)
                print("Attendance Saved ✅")
        elif inside(x,y,buttons["exit"]):
            exit()

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Video Call Attendance System")
cv2.setMouseCallback("Video Call Attendance System", mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    draw_buttons(frame)

    if camera_on:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi,(200,200))
            label, confidence = model.predict(roi)

            name = "Unknown"
            if confidence < 95:
                name = labels[label]
                if name not in attendance:
                    attendance[name] = datetime.now().strftime("%H:%M:%S")

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Video Call Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
