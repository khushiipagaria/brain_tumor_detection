Brain Tumor Classification using CNN
This project is a deep learning-based model for classifying brain MRI images into four categories: glioma tumor, meningioma tumor, no tumor, and pituitary tumor. The model uses a Convolutional Neural Network (CNN) to analyze the MRI images and predict the tumor type.

Project Overview
Goal: Detect and classify brain tumors from MRI images.
Dataset: The project uses the Brain Tumor Classification (MRI) dataset from Kaggle. It includes training and testing images organized into four categories.
Model: A CNN built with Keras and TensorFlow, designed to process and classify images efficiently.
Features
Data Preprocessing:

Resizing images to a uniform size of 150x150 pixels.
Organizing images into input arrays and corresponding labels.
CNN Architecture:

Multiple convolutional layers for feature extraction.
Pooling layers to reduce spatial dimensions.
Dropout layers to prevent overfitting.
Dense (fully connected) layers to make predictions.
Model Training:

Training the CNN on the dataset for 20 epochs with a 90:10 train-validation split.
Categorical cross-entropy as the loss function and Adam optimizer for training.
Evaluation:

Validation data is used to evaluate model accuracy during training.
Dependencies
The following libraries and frameworks are required:

Python
NumPy
Pandas
OpenCV
TensorFlow/Keras
Matplotlib (for visualizations)
Future Scope
Enhance the model using more advanced architectures like ResNet or EfficientNet.
Deploy the model as a web or mobile application for real-time brain tumor detection.
Include additional MRI data for more robust training.
