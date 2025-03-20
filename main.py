##HandTyperMain - Main Class for the HandTyper Programme
##Author: Roel-Junior Alejo Viernes (001221190)
##Email: rv6049z@gre.ac.uk


#import Modules

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import os
model_path = "models/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

result_gesture = None

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Function to print the details of the recognised gesture
    """
    global result_gesture
    if result.gestures:
        print(f"Recognized gestures: {result.gestures}")
        if result.gestures[-1] is not None:
            target = result.gestures[-1][0]
            print(f"Recognized gesture: {target.category_name}")
            result_gesture = str(target.category_name)
            

options = GestureRecognizerOptions(#Parameters for the gesture recognizer
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
    )

#Defining the functions for the hand landmark capture
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main_capture():
    """
    Function to cover the capturing of the frames from cv2
    """
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1) # flip the frame horizontally
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create an image object for Mediapipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) # get the timestamp of the frame
                recognizer.recognize_async(mp_image, frame_timestamp_ms) # recognize the image asynchronously
                results = hands.process(rgb_frame)

                # Draw hand landmarks
                if results.multi_hand_landmarks: # if there are hands in the frame
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # present the hand landmarks onto the frame
                        
                cv2.putText(frame, result_gesture, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # Display the frame
                cv2.imshow('HandTyperv1', frame)            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        cap.release()
        cv2.destroyAllWindows()
        
while __name__ == '__main__':
    live_capture()