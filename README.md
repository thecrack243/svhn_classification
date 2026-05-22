# SVHN Image Classification using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digit images from the SVHN (Street View House Numbers) dataset.

The model is trained to recognize digits from real-world street images and performs multi-class classification for digits from 0 to 9.

---

## Dataset

**Dataset:** SVHN (Street View House Numbers)

The dataset contains RGB images of house numbers collected from Google Street View images.

### Dataset Information
- 73,257 training images
- 26,032 testing images
- Image size: 32 × 32 pixels
- 10 digit classes (0–9)

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- h5py

---

## Project Workflow

1. Load and preprocess the SVHN dataset
2. Normalize image data
3. Convert labels using one-hot encoding
4. Build a Convolutional Neural Network (CNN)
5. Train the model using data augmentation
6. Evaluate performance using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
7. Visualize:
   - Training curves
   - Confusion matrix
   - Correct and incorrect predictions

---

## CNN Architecture

The CNN model consists of:

- Convolutional layers
- ReLU activation
- Max pooling layers
- Fully connected dense layer
- Dropout regularization
- Softmax output layer

---

## Data Augmentation

To improve generalization and reduce overfitting, image augmentation techniques were applied during training, including:

- Rotation
- Width shifting
- Height shifting
- Zooming

---

## Results

The trained CNN model achieved strong classification performance on the SVHN test dataset.

Evaluation metrics include:
- Test Accuracy
- Precision
- Recall
- F1-Score

Additional visualizations such as confusion matrices and prediction samples were generated to analyze model behavior.

---

## Project Structure

```text
project/
│
│
├── img/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── correctly_classified_samples.png
│   ├── misclassified_samples.png
│   ├── svhn_dataset.png
│   └── cnn_diagram.png
│
├── models/
│   └── svhn_model.h5
│
├── notebook/
│   └── svhn.ipynb
│
├── .gitignore
│
├── README.md
│
└── requirements.txt