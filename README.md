DRIVER DROWSINESS AND DISTRACTION DETECTION USING DEEP LEARNING

This project implements a deep learning–based Driver Safety Monitoring System to detect driver drowsiness and distraction in real time using webcam video input. The system uses Convolutional Neural Network (CNN) models trained on eye, yawn, and distraction datasets to identify unsafe driving behavior and improve road safety.

Features

-Real-time driver monitoring using webcam

-Eye closure detection for drowsiness analysis

-Yawn detection using facial features

-Driver distraction detection

-Deep Learning–based classification using CNN

-Real-time alert system for unsafe behavior

-Separate training modules for each detection model

-Trained models saved in .h5 format

Project Structure

Driver-Drowsiness-Distraction-Detection/
├── main.py                     # Main application file
├── train_eye.py                # Eye detection model training
├── train_yawn.py               # Yawn detection model training
├── train_distraction.py        # Distraction detection model training
├── eye_model.h5                # Trained eye detection model
├── yawn_model.h5               # Trained yawn detection model
├── distraction_model.h5        # Trained distraction model
└── README.md                   # Project documentation

Datasets Used
Eye Dataset

Used to classify open and closed eyes for drowsiness detection.

Yawn Dataset

Used to identify yawning behavior.

Distraction Dataset

Used to detect distracted driving activities.

Technologies Used

-Python

-OpenCV

-TensorFlow

-Keras

-NumPy

-CNN (Convolutional Neural Network)

Requirements

-Python 3.10+

-opencv-python

-tensorflow

-keras

-numpy

Working Principle

1.Webcam captures the driver’s face continuously.

2.Facial features are detected using OpenCV.

3.CNN models analyze:

-Eye status

-Yawning behavior

-Driver distraction

4.The system predicts whether the driver is attentive or drowsy.

5.Alerts are generated when unsafe behavior is detected.

Results
-Real-time detection of drowsiness and distraction

-Accurate classification using CNN models

-Improves driver safety and accident prevention

-Suitable for smart vehicle monitoring systems

Future Work
-Add voice alarm integration

-Mobile application support

-Night vision detection

-Cloud-based monitoring system

-GPS emergency alert integration

-Improve model accuracy with larger datasets







Author

Shifana N  
GitHub: https://github.com/Shifana4303


