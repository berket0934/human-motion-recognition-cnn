# Human Motion Recognition Using IMU Sensors

Deep learning-based human activity recognition using multichannel IMU sensor data and a 1D Convolutional Neural Network.

## Project Overview

This project classifies human activities from acceleration data collected using IMU sensors placed on the lower back and thigh.

The preprocessing pipeline includes:
- Sliding-window segmentation
- 50% overlap between windows
- Z-score normalization
- Time-series feature learning with a 1D CNN

## Model Architecture

The model uses multiple 1D convolutional layers with increasing filter depth:

- Conv1D: 32 filters
- Conv1D: 64 filters
- Conv1D: 128 filters
- Batch Normalization
- Max Pooling
- Dropout
- Global Average Pooling
- Softmax Classification

## Results

- Test Accuracy: **96.74%**
- Average F1 Score: **0.92**

Activity-level performance:

| Activity | Accuracy |
|---|---:|
| Lying | 100% |
| Sitting | 98.38% |
| Standing | 93.75% |
| Walking | 94.74% |
| Running | 97.21% |

## Technologies

- MATLAB
- Simulink
- Deep Learning
- 1D CNN
- IMU Sensors
- Time-Series Analysis
- Signal Processing

## Future Work

Future improvements may include:
- Real-time inference using live IMU streams
- Larger and more diverse datasets
- Deployment to embedded or wearable systems

## Author

Berke Tüylek
