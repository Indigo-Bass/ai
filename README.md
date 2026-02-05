Speech Emotion Recognition (SER) - AI Club Task

AVIRAL SINGH
2025A7PS0603P
BITS Pilani, Pilani Campus

Project Overview
This project implements a Robust Deep Learning model to classify human emotions from audio files (.wav). The system utilizes a 2D Convolutional Neural Network (CNN) trained on the RAVDESS dataset, achieving high generalization through advanced audio augmentation and class-balancing techniques.

Model Performance
The model was evaluated on a held-out test set (15% of total data), yielding the following results:
Overall Accuracy: 89%
Macro F1-Score: 0.89
Pitch Bias Check:
  Male Accuracy: 87.96%
  Female Accuracy: 89.81%
  
Training Details
Data Augmentation
To improve model robustness and prevent overfitting, the training data was tripled using:
Noise Injection: Added white noise to simulate real-world environments.
Pitch Shifting: Altered pitch by 0.7 steps to neutralize gender-based bias.
Time Stretching: Randomly varied speed (0.8x to 1.2x) to handle different speaking rates.

Model Architecture (2D CNN)
Input: 128x128 Mel-Spectrograms.
Layers: 3 Convolutional blocks with Batch Normalization for faster convergence and Global Average Pooling to reduce parameter count.
Regularization: Dropout (0.4) to prevent memorization of training samples.

Key Settings & Optimization
Class Balancing: Implemented class_weight (2.0 for Neutral) to compensate for data imbalance.
EarlyStopping: Automatically halted training when validation loss stopped improving (Patience: 8).
Learning Rate Scheduler: Used ReduceLROnPlateau to fine-tune weights during plateaus.

How to Run:
This project supports one-command execution for easy evaluation.
Install Requirements:
    pip install -r requirements.txt
Run Prediction:
    python predict.py "path/to/your/audio.wav"
