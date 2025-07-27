# Objective

This project detects whether people are wearing a face mask or not using a Convolutional Neural Network (CNN) trained on image data, and performs **real-time face mask detection** through webcam with audio alerts.

## Features

- Trainable CNN model using Keras and TensorFlow
- Real-time face detection using Haar Cascades
- Classification of face as **"with mask"** or **"without mask"**
- Alarm sound when a person is detected **without mask**
- Live webcam preview with bounding boxes and labels


## 🛠️ Tools & Libraries Used

- Python 3.x
- TensorFlow / Keras
- OpenCV
- scikit-learn
- pygame (for alarm sound)
- Haarcascade XML for face detection

# Step of How to run

## 🗂️ Project Structure

face-mask-detection/
├── train.py # Model training script
├── test.py # Real-time detection with webcam
├── alarm.mp3 # Alarm sound file (you must provide)
├── model2-010.keras # Trained model file
├── haarcascade_frontalface_default.xml # Face detector
├── Dataset/
│ ├── train/
│ │ ├── with_mask/
│ │ └── without_mask/
│ └── test/
│ ├── with_mask/
│ └── without_mask/

## Train the Model

-> python train.py

##  Run Real-Time Mask Detection

-> python test.py









