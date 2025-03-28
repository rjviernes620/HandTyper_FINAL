##capture.py
##Author: Roel-Junior Alejo Viernes (001221190)
##Email: rv6049z@gre.ac.uk
#Description: This script is to capture relevant frames for forthcoming gestures to add to the dataset

import cv2
import os

cap = cv2.VideoCapture(0)

# Create a directory for the dataset
if not os.path.exists('HandTyper_FINAL/data'):
    os.makedirs('HandTyper_FINAL/data')
    
# Begin capturing the frames
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # flip the frame horizontally
    cv2.imshow('frame', frame)
    cv2.imwrite('HandTyper_FINAL/data/frame.jpg', frame)
    
    # Press 'q' to quit the capturing process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

