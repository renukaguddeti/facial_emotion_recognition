import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import os
import cv2
import numpy as np

# ✅ Define Emotion Labels
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ✅ Get Absolute Path for Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src folder
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "emotion_model.h5")

# ✅ Load Model with `.h5` Format Support
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

print("🚀 Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)  # Force HDF5 format loading

# ✅ Load Haarcascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Start Webcam for Real-Time Emotion Detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to model input size
        roi_gray = roi_gray.astype("float32") / 255.0  # Normalize
        roi_gray = np.expand_dims(roi_gray, axis=[0, -1])  # Reshape for model

        prediction = model.predict(roi_gray)
        emotion_index = np.argmax(prediction)  # Get predicted class index
        emotion = EMOTION_LABELS[emotion_index]  # Convert index to emotion name

        # Draw rectangle and label with Emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
