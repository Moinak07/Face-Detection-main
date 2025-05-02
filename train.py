import os
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load data
data_dir = os.path.join(os.getcwd(), 'Clean Data')
with open(os.path.join(data_dir, "Images.p"), 'rb') as f:
    X = pickle.load(f)
with open(os.path.join(data_dir, "labels.p"), 'rb') as f:
    y = pickle.load(f)

# Convert list of images to numpy array
X = np.array(X)

# Prepare data
X = X.reshape(-1, 100, 100, 1).astype('float32') / 255.0

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder for later use in recognition
with open('label_encoder.p', 'wb') as f:
    pickle.dump(le, f)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_categorical, epochs=10, batch_size=8, verbose=1)

# Save the trained model
model.save('face_recognition_model.h5')

print("Training complete. Model and label encoder saved.")