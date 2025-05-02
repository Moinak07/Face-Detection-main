# Face Detection and Recognition

This project is a Python-based face recognition system that allows you to collect face images, train a deep learning model, and recognize people in real time using your laptop's webcam.

## Features
- Data Collection: Collect face images for any number of people using your phone's IP Webcam app or a webcam.
- Flexible Dataset: Add new people at any time. All data is appended automatically.
- Model Training: Train a Convolutional Neural Network (CNN) on your collected data.
- Real-Time Recognition: Recognize faces live using your webcam.
- Easy Reset: Delete all collected data to start a new dataset without changing any code.


## Project Structure
Face-Detection-main/
│
├── collect_data.py        # Script to collect and append face images and labels
├── train.py               # Script to train the CNN model
├── recognize.py           # Script for real-time face recognition
├── Clean Data/            # Folder where images and labels are stored
│   ├── Images.p
│   └── labels.p
├── face_recognition_model.h5  # Trained model (generated after training)
├── label_encoder.p            # Label encoder (generated after training)
└── haarcascade_frontalface_default.xml #Haar Cascade file for face detection                     



## Requirements
- Python 3.x
- OpenCV (opencv-python)
- NumPy
- Keras
- TensorFlow
- scikit-learn
- Pickle (part of Python standard library)

## Install All Required Libraries
- pip install opencv-python numpy keras tensorflow scikit-learn
- pip install matplotlib (optional)

## How to Use
1) Collect Face Data 
Install "IP Webcam" on your phone and connect it to your computer.
Run the collect_data.py script to collect face data from a video stream.

Enter your IP Webcam URL (e.g., http://100.165.30.213:8000/shot.jpg) or use your webcam.
The script will collect images for one person at a time.
Enter the person's name when prompted.
Repeat for each new person. Data is appended automatically.

2) Train the Model
Run the train.py script to train the CNN model on the collected data.
This will train a CNN model and save it as face_recognition_model.h5 along with the label encoder.

3) Recognize Faces
Run the recognize.py script to recognize faces in real-time using the trained model.
The webcam will open and display the recognized person's name above their face.
If the model is not confident, it will display "Unknown".

4) Reset Data
To start a new data collection (delete all previous records):

Delete the files Clean Data/Images.p 
                 Clean Data/labels.p
Run collect_data.py again to begin collecting new data.




