from mediapipe.tasks.python import vision
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load the exported .task model
model_path = "evaluate\gesture_recognizer.task"  # Replace with your .task file path

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

options = GestureRecognizerOptions(#Parameters for the gesture recognizer
    base_options=BaseOptions(model_asset_path=model_path),
    )

gesture_recognizer = GestureRecognizer.create_from_options(options)

test_images = []  # List to store test images
test_labels = []  # List to store corresponding labels

labels = []

dataset_path = ["\wsl.localhost\Ubuntu\\root\HandTyper_MAIN\1_HAND_DATASET", "\\wsl.localhost\Ubuntu\root\HandTyper_MAIN\2_HAND_DATASET"]

for folder in dataset_path:
    for data in os.listdir(folder):
        labels.append(data)
    
# Prepare test data

for dataset in dataset_path:
    for label in labels:  # Assuming `labels` contains class names
        label_dir = os.path.join(dataset, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            image = mp.Image.create_from_file(file_path)  # Load image using MediaPipe
            test_images.append(image)
            test_labels.append(label)
            
            
        predicted_labels = []

        for image in test_images:
            result = gesture_recognizer.recognize(image)
            if result.gestures:  # If a gesture is recognized
                predicted_labels.append(result.gestures[0][0].category_name)  # Take the top prediction
            else:
                predicted_labels.append("Unknown")  # Handle cases where no gesture is recognized
                
                

        # Convert labels to numpy arrays
        true_labels = np.array(test_labels)
        predicted_labels = np.array(predicted_labels)
        
       # set = logistic_regression(predicted_labels, true_labels)

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
