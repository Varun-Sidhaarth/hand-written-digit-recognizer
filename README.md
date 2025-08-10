# Handwritten Digit Recognizer

This project implements a neural network-based handwritten digit recognizer using the MNIST dataset. The model can accurately predict handwritten digits from 0 to 9.

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `digit_recognizer.py`: Main script for training the model
- `test_digit.py`: Script for testing the model with custom handwritten digits
- `requirements.txt`: List of required Python packages
- `digit_recognizer_model.h5`: Trained model (generated after training)
- `training_history.png`: Training history visualization (generated after training)

## Usage

1. Train the model:
```bash
python digit_recognizer.py
```
This will:
- Load the MNIST dataset
- Train the neural network model
- Save the trained model as 'digit_recognizer_model.h5'
- Generate a training history plot

2. Test with custom handwritten digits:
```bash
python test_digit.py
```
Note: Before running the test script, make sure to:
- Replace 'your_digit_image.png' in test_digit.py with the path to your handwritten digit image
- The input image should be a clear, single digit
- The digit should be written in black on a white background
- The image will be automatically resized to 28x28 pixels

## Model Architecture

The model uses a simple neural network with:
- Input layer (784 neurons - flattened 28x28 image)
- Dense layer (128 neurons) with ReLU activation
- Dropout layer (0.2)
- Dense layer (64 neurons) with ReLU activation
- Dropout layer (0.2)
- Output layer (10 neurons) with softmax activation

## Performance

The model typically achieves:
- Training accuracy: >98%
- Validation accuracy: >97%

The exact performance metrics will be shown during training and saved in the training history plot. 