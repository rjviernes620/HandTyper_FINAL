##HandTyperMain - Main Class for the HandTyper Programme
##Author: Roel-Junior Alejo Viernes (001221190)
##Email: rv6049z@gre.ac.uk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import tkinter as tk
import os
from tkinter import messagebox

import pynput
import threading

model_path = "models/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

result_gesture = None

key_mode = False
mouse_mode = False

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Function to print the details of the recognised gesture
    """
    global result_gesture
    if result.gestures:
        
        while key_mode == True:
            print(f"Recognized gestures: {result.gestures}")
            target = result.gestures[0][0].category_name
            
            if target != "":
                result_gesture = result.gestures[0][0].category_name  # Access the category_name attribute
                translate(result_gesture)  # Translate the gesture to a keypress
                
            else:
                result_gesture = "No gesture detected"
        
        while mouse_mode == True:
            
            
            pass
            

options = GestureRecognizerOptions(#Parameters for the gesture recognizer
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
    )

#Defining the functions for the hand landmark capture
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def draw_menu(frame):
    """
    Function to draw a simple menu overlay on the video feed.
    """
    # Draw a rectangle for the menu background
    cv2.rectangle(frame, (0, 0), (320, 50), (200, 200, 200), -1)  # Light gray background

    # Add menu options as text
    cv2.putText(frame, "Menu: [Q] Quit | [H] Help", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def video_feed():
    cv2.namedWindow('HandTyperv1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HandTyperv1', windowsize()[0], windowsize()[1])

    with mp_hands.Hands(static_image_mode = False, max_num_hands = 2, min_detection_confidence = 0.75) as hands:
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Set desired FPS
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) # flip the frame horizontally
            
            frame = cv2.resize(frame, (1280, 960)) # resize the frame to the size of the window
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #Create an image object for Mediapipe
            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) # get the timestamp of the frame
            #recognizer.recognize_async(mp_image, frame_timestamp_ms) # recognize the image asynchronously
            results = hands.process(frame)

            # Draw hand landmarks
            if results.multi_hand_landmarks: # if there are hands in the frame
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x = int(index_finger_tip.x * frame.shape[1])
                    y = int(index_finger_tip.y * frame.shape[0])
                    
                    mouse_movement(x, y) # move the mouse to the x and y coordinates
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                    
            cv2.imshow('HandTyperv1', frame)   
            #cv2.moveWindow('HandTyperv1', (windowsize()[0] - 330), (windowsize()[1] - 250))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit the application
                break
            
            elif key == ord('m'):  # Show help message
                print("Placeholder Text")
                
    
        
        
def mouse_movement(x, y):
    """
    Scale the coordinates from the frame resolution to the screen resolution
    and move the mouse to the scaled position.
    """
    screen_width, screen_height = 1920, 1080  # Get the screen dimensions
    frame_width, frame_height = 1280, 960  # Replace with your frame resolution

    # Scale the coordinates
    scaled_x = int(x * screen_width / frame_width)
    scaled_y = int(y * screen_height / frame_height)

    # Move the mouse to the scaled coordinates
    pynput.mouse.Controller().position = (scaled_x, scaled_y)# move the mouse to the x and y coordinates
        
def windowsize():
    main = tk.Tk()
    return main.winfo_screenwidth() // 4, main.winfo_screenheight() // 4    

def translate(sign):
    keyboard = pynput.keyboard.Controller()
    target = sign.split()
    keyboard.press(target[-1]) # press the key corresponding to the gesture
    
    if target == "BACKSPACE":
        keyboard.press(pynput.keyboard.Key.backspace)
        keyboard.release(pynput.keyboard.Key.backspace)
    
    elif target == "SPACE":
        keyboard.press(pynput.keyboard.Key.space)
        keyboard.release(pynput.keyboard.Key.space)

    elif target == "ENTER":
        keyboard.press(pynput.keyboard.Key.enter)
        keyboard.release(pynput.keyboard.Key.enter)
    
    print(f"Typed: {target[-1]}")

        
while __name__ == '__main__':
    video_feed()