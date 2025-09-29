# Face & Emotion Recognition with TensorFlow and OpenCV

This project is a from-scratch implementation of a basic **Face Recognition** and **Emotion Detection** system using Python, TensorFlow, and OpenCV.

---

## Features
- **Custom Face Recognition**: Train the model on your own dataset of faces (no pre-downloaded datasets).  
- **Emotion Detection**: Classify faces into emotions such as **Happy, Sad, Neutral, and Surprised**.  
- **Webcam Integration**: Capture and label training images directly from your webcam.  
- **Lightweight Model**: Small and fast to train, focused on learning the fundamentals.  
- **Educational Focus**: Demonstrates the machine learning pipeline rather than competing with production-grade systems.  

---

## Getting Started

### 1. Clone the Repository

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Dependencies:
1. tensorflow
2. opencv-python
3. numpy
4. matplotlib

### 4. Capture Training Data
Use the webcam capture script to collect images for each person/emotion.
(Note: the data/ folder is git-ignored, so your personal images won’t be uploaded.)

Example:
```bash
python capture.py --type person --label NAME --count 200
python capture.py --type emotion --label EMOTION --count 200
```

### 5. Train the Model
```bash
python train_face_recognizer.py
python train_emotion_cnn.py
```

### 6. Run Real-Time Detection
```bash
python realtime_demo.py
```

## Project Structure

```bash
face-emotion-project/
├─ capture.py
├─ train_face_recognizer.py
├─ train_emotion_cnn.py
├─ realtime_demo.py
├─ requirements.txt
├─ models/
└─ data/
   ├─ faces/            # for face recognition: data/faces/<person_name>/<img>.jpg
   └─ emotions/         # for emotion training: data/emotions/<emotion_label>/<img>.jpg
```