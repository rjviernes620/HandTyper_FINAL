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

import threading
import pynput

model_path = "models/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

result_gesture = None

key_mode = False
mouse_mode = True

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
        
        # while mouse_mode == True:
            
        #     mouse_movement(result)
            
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


stop_thread = False

def video_capture(recognizer, hands):
    
    global stop_thread
    
    cap = cv2.VideoCapture(0)
    frame_count = 0  # Initialize frame count
    
    while not stop_thread:
        
        ret, frame = cap.read()
        cv2.imshow('HandTyperv1', frame)

        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 2 == 0:  # Process every other frame
            continue
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        # Zoom out by resizing the frame to a smaller resolution
        frame = cv2.resize(frame, (int(windowsize()[0] // 4), int(windowsize()[-1] // 4)), interpolation=cv2.INTER_NEAREST)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  # Create an image object for Mediapipe
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))  # Get the timestamp of the frame
        #recognizer.recognize_async(mp_image, frame_timestamp_ms)  # Recognize the image asynchronously
        
        # Draw the menu overlay
        draw_menu(frame)
        cv2.putText(frame, result_gesture, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

        # Display the frame
        cv2.resizeWindow('HandTyperv1', windowsize()[0] // 4, windowsize()[1] // 4)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the application
            stop_thread = True
            break

    cap.release()
    cv2.destroyAllWindows()
        
def main_capture():
    
    global stop_thread
    with GestureRecognizer.create_from_options(options) as recognizer:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            video_feed = threading.Thread(target=video_capture, args=(recognizer, hands))
            video_feed.start()
            
            video_feed.join()  # Wait for the thread to finish before exiting
        
        
def mouse_movement(x, y):
    
    pynput.mouse.Controller().position = (x, y)  # Move the mouse to the specified coordinates

def windowsize():
    """
    Function to get the size of the screen
    """
    
    main = tk.Tk()
    return main.winfo_screenwidth(), main.winfo_screenheight()    

def translate(sign):
    """
    Main Function, translating the identified gesture into a keystroke
    """
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
    main_capture()