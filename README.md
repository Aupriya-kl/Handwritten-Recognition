Handwritten Digit Recognition using Deep Learning
This project implements a Handwritten Digit Recognition (HWR) system using Convolutional Neural Networks (CNNs) on the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), where each image is 28x28 pixels. The model is trained to classify these digits with high accuracy.

Technologies Used
Python
TensorFlow & Keras (for deep learning)
NumPy & Pandas (for data processing)
Matplotlib (for visualization)
Google Colab (for training the model)
Project Workflow
Load Data: Download the MNIST dataset.
Preprocessing: Normalize pixel values and reshape the data for CNN input.
Model Architecture:
Conv2D & MaxPooling layers for feature extraction.
Flatten & Dense layers for classification.
Training: Train the CNN model for 5 epochs using Adam optimizer.
Evaluation:
Plot training and validation accuracy.
Evaluate the model on test data.
Prediction & Visualization: Display sample test images with predicted labels.
Saving Model: The trained model (mnist_model.h5) and training history (training_history.pickle) are saved for future use.
