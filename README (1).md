# Face Detection and Recognition

This project implements a face detection and recognition system using OpenCV and a pre-trained Haar Cascade classifier. It collects face data, processes it, and uses a trained model to recognize individuals.

## Project Structure
FaceRecognitionProject

├── consolidated_data.py              
├── final_model.h5                    
├── haarcascade_frontalface_default.xml    
├── recognize.py                  
├── README.md                         



## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pickle

Install the required libraries using:

```bash
pip install opencv-python numpy matplotlib


How to Use
1. Collect Face Data
Run the collect_data.py script to collect face data from a video stream.
python [collect_data.py](http://_vscodecontentref_/7)


The script captures 100 face images from the video stream.
Enter the name of the person when prompted. The images will be saved in the Images/ directory.
2. Consolidate Data
Run the consolidated_data.py script to preprocess and consolidate the collected data.
python [consolidated_data.py](http://_vscodecontentref_/8)

The script resizes, converts to grayscale, and stores the images and labels in the Clean Data/ directory as .p files.
3. Recognize Faces
Run the recognize.py script to recognize faces using the pre-trained model.

python [recognize.py](http://_vscodecontentref_/9)


The script uses the final_model.h5 file to predict the identity of the person in the video stream.

Key Features
Face Detection: Uses Haar Cascade to detect faces in real-time.
Data Preprocessing: Resizes, converts to grayscale, and normalizes images for model training.
Face Recognition: Recognizes faces using a pre-trained deep learning model.


File Descriptions
collect_data.py: Captures face images from a video stream and saves them with labels.
consolidated_data.py: Preprocesses the collected images and stores them for training.
recognize.py: Recognizes faces using the pre-trained model.
haarcascade_frontalface_default.xml: Haar Cascade XML file for face detection.
final_model.h5: Pre-trained model for face recognition.


License
This project is licensed under the MIT License.
