import cv2
import urllib.request
import numpy as np
import pickle
import os

# Path to Haar Cascade classifier
classifier = cv2.CascadeClassifier(r"C:\Users\KIIT\OneDrive\Desktop\Face Detection\Face Detection\haarcascade_frontalface_default.xml")

# Ask for the IP Webcam URL
url = input("Enter IP Webcam URL (e.g., http://192.168.29.213:8080/shot.jpg): ")

data = []
max_images = 100

print("Collecting data... Press 'q' to quit early.")

while True:
    # Fetch image from the IP Webcam stream
    image_from_url = urllib.request.urlopen(url)
    img_array = np.array(bytearray(image_from_url.read()), np.uint8)
    frame = cv2.imdecode(img_array, -1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        data.append(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Face', face)
        print(f"{len(data)} /{max_images}")
        if len(data) >= max_images:
            break

    cv2.putText(frame, str(len(data)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == ord("q") or len(data) >= max_images:
        break

cv2.destroyAllWindows()

if len(data) == max_images:
    name = input("Enter Face holder name : ")
    data_dir = "Clean Data"
    os.makedirs(data_dir, exist_ok=True)

    # Load existing data if present
    images_path = os.path.join(data_dir, "Images.p")
    labels_path = os.path.join(data_dir, "labels.p")
    if os.path.exists(images_path) and os.path.exists(labels_path):
        with open(images_path, "rb") as f:
            all_images = pickle.load(f)
        with open(labels_path, "rb") as f:
            all_labels = pickle.load(f)
    else:
        all_images = []
        all_labels = []

    all_images.extend(data)
    all_labels.extend([name] * len(data))

    with open(images_path, "wb") as f:
        pickle.dump(all_images, f)
    with open(labels_path, "wb") as f:
        pickle.dump(all_labels, f)
    print("Data appended and saved.")
else:
    print("Need more data")