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
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

result_gesture = None
global mouse_mode, key_mode

key_mode = False
mouse_mode = True

model_path = "Hand_Data_V7/gesture_recognizer.task" # Path to the model file
    
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Function to print the details of the recognised gesture
    
    Params:
    result: The result of the gesture recognition (GestureRecognizerResult)
    output_image: The output image with the recognised gesture (mp.Image)
    timestamp_ms: The timestamp of the frame (int)
    
    """
    global result_gesture
    if result.gestures:
        
        print(f"Recognized gestures: {result.gestures}")
        target = str(result.gestures[0][0].category_name)
        
        if target != '' and target != None and target != 'none':
            result_gesture = result.gestures[0][0].category_name  # Access the category_name attribute
            translate(result_gesture)  # Translate the gesture to a keypress
            
        else:
            result_gesture = "No gesture detected"            

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
    
    Params:
    frame: The current video frame from OpenCV.
    """
    # Draw a rectangle for the menu background
    cv2.rectangle(frame, (0, 0), (320, 50), (200, 200, 200), -1)  # Light gray background

    # Add menu options as text
    cv2.putText(frame, "Menu: [Q] Quit | [H] Help", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main_capture():
    """
    Function to cover the capturing of the frames from cv2
    """
    cv2.namedWindow('HandTyperv1', cv2.WINDOW_NORMAL) # Create a window for the video feed
    cv2.resizeWindow('HandTyperv1', windowsize()[0], windowsize()[1]) # Resize the window to fit the screen size
    cv2.setWindowProperty('HandTyperv1', cv2.WND_PROP_TOPMOST, 1)  # Set the window to always be on top

    with GestureRecognizer.create_from_options(options) as recognizer:
        
            
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.75) as hands:
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)  # Flip the frame horizontally

                frame = cv2.resize(frame, (windowsize()[0], windowsize()[1])) # Resize the frame to fit the window size
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) #Create an image object for Mediapipe
                frame_timestamp_ms = int(time.time() * 1000) # get the timestamp of the frame
                recognizer.recognize_async(mp_image, frame_timestamp_ms) # recognize the image asynchronously
                results = hands.process(rgb_frame) 

                # Draw hand landmarks
                if results.multi_hand_landmarks: # if there are hands in the frame
                    if mouse_mode == True:
                        hand_landmarks = results.multi_hand_landmarks[0] # get the first hand landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # present the hand landmarks onto the frame

                        if mouse_mode:
                            
                            mouse = pynput.mouse.Controller()

                            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            index_x = int(index_finger_tip.x * frame.shape[1]) 
                            index_y = int(index_finger_tip.y * frame.shape[0])
                            
                            mouse_movement(index_x, index_y) # move the mouse to the index finger tip position
                            
                            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            middle_x = int(middle_finger_tip.x * frame.shape[1])
                            middle_y = int(middle_finger_tip.y * frame.shape[0])
                            
                            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1) # draw a circle around the index finger tip
                            
                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            thumb_x = int(thumb_tip.x * frame.shape[1])
                            thumb_y = int(thumb_tip.y * frame.shape[0])
                            
                            thumb_index_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5 # calculate the distance between the index finger and thumb
                            mid_index_distance = ((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2) ** 0.5 # calculate the distance between the middle finger and index finger
                            
                            print(f"Thumb distance: {thumb_index_distance}, Middle distance: {mid_index_distance}") # print the distances for debugging
                            
                            if thumb_index_distance < 25: # if the thumb and index finger are close together
                                mouse.click(pynput.mouse.Button.right, 1) # click the left mouse button    
                                
                            elif mid_index_distance < 25: # if the middle finger and index finger are close together
                                mouse.click(pynput.mouse.Button.left, 1) # click the right mouse button                            
                                                                    
                # Draw the menu overlay
                draw_menu(frame)
                        
                cv2.putText(frame, result_gesture, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                # Display the frame
                cv2.imshow('HandTyperv1', frame)   
                #cv2.moveWindow('HandTyperv1', (windowsize()[0] - 330), (windowsize()[1] - 250))

                global key_mode, mouse_mode
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit the application
                    break
                
                elif key == ord('m'):  # Show help message
                    if mouse_mode != True:
                        mouse_mode = True
                        key_mode = False
                    else:
                        mouse_mode = False
                        key_mode = True
                
        cap.release()
        cv2.destroyAllWindows()
        
def windowsize():
    """
    Function to get the screen size of the relevant PC
    
    Returns:
    Tuple of width and height of the screen
    """
    main = tk.Tk()
    return main.winfo_screenwidth() // 4, main.winfo_screenheight() // 4          

def mouse_movement(x, y):
    
    """
    Function to move the mouse to the x and y coordinates of the index finger tip
    
    Params:
    x: x coordinate of the index finger tip
    y: y coordinate of the index finger tip
    """
    mouse = pynput.mouse.Controller()
    current_x, current_y = mouse.position  # Get the current mouse position


    scaled_x = int(x * int(windowsize()[0] * 4) / int(windowsize()[1] * 4))
    scaled_y = int(y * int(windowsize()[1] * 4) / int(windowsize()[0] * 4))# Scale the x coordinate to the screen resolution
    # Number of steps for smooth movement
    steps = 10
    for i in range(1, steps + 1):
        # Interpolate between the current position and the target position
        smooth_x = current_x + (scaled_x - current_x) * i / steps
        smooth_y = current_y + (scaled_y - current_y) * i / steps
        mouse.position = (int(smooth_x), int(smooth_y))
        time.sleep(0.01)  # Add a small delay for smoother movement
        
  

def translate(sign):
    
    """
    Function to translate the recognised gesture to a keypress
    
    Params:
    sign: the recognised gesture (str)
    """
    keyboard = pynput.keyboard.Controller()
    mouse = pynput.mouse.Controller()
    target = sign.split()
    
    if target == "BACKSPACE":
        keyboard.press(pynput.keyboard.Key.backspace)
        keyboard.release(pynput.keyboard.Key.backspace)
    
    elif target == "SPACE": 
        keyboard.press(pynput.keyboard.Key.space)
        keyboard.release(pynput.keyboard.Key.space)

    elif target == "ENTER":
        keyboard.press(pynput.keyboard.Key.enter)
        keyboard.release(pynput.keyboard.Key.enter)
        
    elif sign == "CLICK":
        mouse.click(pynput.mouse.Button.left, 1) # click the left mouse button
        
    elif sign == "RIGHT_CLICK":
        mouse.click(pynput.mouse.Button.right, 1)
    
    elif target == '10':
        keyboard.press(pynput.keyboard.KeyCode.from_char('1'))
        keyboard.release(pynput.keyboard.KeyCode.from_char('1'))
        keyboard.press(pynput.keyboard.KeyCode.from_char('0'))
        keyboard.release(pynput.keyboard.KeyCode.from_char('0'))
    
    else:
        keyboard.press(target[-1]) # press the key corresponding to the gesture
    print(f"Typed: {target[-1]}")

        
while __name__ == '__main__':
    main_capture()