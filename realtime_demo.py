# realtime_demo.py
import cv2
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf

MODEL_DIR = Path('models')
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load LBPH model and label map
lbph_path = MODEL_DIR/'lbph_face_model.xml'
label_map_path = MODEL_DIR/'label_map.pkl'
if not lbph_path.exists() or not label_map_path.exists():
    raise SystemExit("LBPH model or label map not found. Run train_face_recognizer.py first.")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(lbph_path))
with open(label_map_path,'rb') as f:
    label_map = pickle.load(f)
inv_label_map = {v:k for k,v in label_map.items()}

# Load emotion CNN
emotion_model_path = MODEL_DIR/'emotion_cnn.h5'
if not emotion_model_path.exists():
    raise SystemExit("Emotion model not found. Run train_emotion_cnn.py first.")
emotion_model = tf.keras.models.load_model(str(emotion_model_path))

# emotion classes come from training generator class indices mapping;
# assume you saved the mapping by hand or inspect folder order. For simplicity, we reconstruct:
emotion_classes = list(sorted([d.name for d in (Path('data/emotions')).iterdir() if d.is_dir()]))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    for (x,y,w,h) in faces:
        face_img = frame_gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (150,150))
        # Identity prediction (LBPH)
        try:
            label_id, conf = recognizer.predict(face_resized)
            name = inv_label_map.get(label_id, 'Unknown')
            id_text = f"{name} ({conf:.1f})"
        except Exception as e:
            id_text = "Unknown"
        # Emotion prediction (CNN)
        emo_input = cv2.resize(face_img, (64,64))
        emo_input = emo_input.astype('float32') / 255.0
        emo_input = np.expand_dims(emo_input, axis=(0,-1))  # shape (1,64,64,1)
        preds = emotion_model.predict(emo_input)
        emo_idx = int(np.argmax(preds))
        emo_label = emotion_classes[emo_idx] if emo_idx < len(emotion_classes) else 'unknown'
        conf_emo = preds[0][emo_idx]
        # Draw
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, id_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"{emo_label} {conf_emo:.2f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow('Face + Emotion', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
