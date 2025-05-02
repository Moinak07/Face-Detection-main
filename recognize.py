import cv2
import numpy as np
from keras.models import load_model
import pickle

# Load the trained model and label encoder
model = load_model('face_recognition_model.h5')
with open('label_encoder.p', 'rb') as f:
    le = pickle.load(f)

# Load Haar Cascade classifier for face detection
classifier = cv2.CascadeClassifier(r"C:\Users\KIIT\OneDrive\Desktop\Face Detection\Face Detection\haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

def get_pred_label(label_idx):
    # Handle out-of-bounds
    if label_idx >= len(le.classes_):
        print(f"Predicted index {label_idx} out of bounds, defaulting to 0.")
        label_idx = 0
    return le.inverse_transform([label_idx])[0]

CONFIDENCE_THRESHOLD = 0.5  # You can adjust this

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face, (100, 100)).reshape(1, 100, 100, 1).astype('float32') / 255.0

        preds = model.predict(face_img)
        label_idx = np.argmax(preds, axis=1)[0]
        confidence = preds[0][label_idx]

        # Debug prints
        print("Model prediction (raw):", preds)
        print("Predicted index:", label_idx)
        print("Known classes in label encoder:", le.classes_)
        print("Predicted name:", le.inverse_transform([label_idx])[0])
        print("Confidence:", confidence)

        if confidence < CONFIDENCE_THRESHOLD:
            display_name = "Unknown"
        else:
            name = get_pred_label(label_idx)
            display_name = name.capitalize()

        cv2.putText(frame, display_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()