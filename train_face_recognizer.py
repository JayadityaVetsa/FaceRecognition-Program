# train_face_recognizer.py
import cv2
import os
from pathlib import Path
import numpy as np
import pickle

DATA_DIR = Path('data/faces')
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(exist_ok=True, parents=True)

def prepare_training_data(data_dir):
    images = []
    labels = []
    label_map = {}
    next_label = 0
    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir(): continue
        name = person_dir.name
        if name not in label_map:
            label_map[name] = next_label
            next_label += 1
        label = label_map[name]
        for img_file in person_dir.glob('*.jpg'):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (150,150))
            images.append(img)
            labels.append(label)
    return images, labels, label_map

images, labels, label_map = prepare_training_data(DATA_DIR)
print("Found classes:", label_map)
if len(images) == 0:
    raise SystemExit("No training images found in data/faces. Run capture.py first.")

# Make sure opencv-contrib is installed for face module
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, np.array(labels))

recognizer.save(str(MODEL_PATH/'lbph_face_model.xml'))
with open(MODEL_PATH/'label_map.pkl','wb') as f:
    pickle.dump(label_map, f)
print("Saved LBPH model and label map to models/")
