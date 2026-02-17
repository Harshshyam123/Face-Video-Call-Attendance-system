# Video Call Attendance System

## Description
A real-time video call attendance system using face recognition.
The system detects and recognizes multiple students and automatically
records attendance in a CSV file.

## Features
- Real-time face detection and recognition
- Multiple student support
- Mouse-based UI buttons inside OpenCV window
- Automatic attendance marking
- CSV export

## Tech Stack
- Python
- OpenCV (LBPH Face Recognition)
- NumPy
- Pandas

## Project Structure
- dataset/ : Student-wise face images
- src/ : Source code
- attendance/ : Generated attendance files

## How to Run
```bash
python src/train.py
python src/main_with_ui_buttons.py
## Note
The attendance CSV file is auto-generated when the application runs.

# Face-Video-Call-Attendance-system
