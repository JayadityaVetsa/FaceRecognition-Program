# capture.py
import cv2
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['person','emotion'], required=True, help='Collect images for a person or an emotion')
parser.add_argument('--label', required=True, help='Name of person or emotion (e.g., "jay", "happy")')
parser.add_argument('--count', type=int, default=200, help='Number of images to capture')
parser.add_argument('--output', default='data', help='Base output folder')
args = parser.parse_args()

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

out_dir = Path(args.output) / ('faces' if args.type=='person' else 'emotions') / args.label
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press SPACE to capture a frame, ESC to quit early.")
count = 0
while count < args.count:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame, f"Captured: {count}/{args.count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow('capture', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # ESC
        break
    if k == 32: # SPACE -> save from first face detected
        if len(faces) == 0:
            print("No face detected — try again.")
            continue
        x,y,w,h = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (150,150))
        path = out_dir / f'{args.label}_{count:03d}.jpg'
        cv2.imwrite(str(path), face_img)
        count += 1
        print("Saved:", path)
cap.release()
cv2.destroyAllWindows()
