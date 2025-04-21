##HandTyperMain - Main Class for the HandTyper Programme
##Author: Roel-Junior Alejo Viernes (001221190)
##Email: rv6049z@gre.ac.uk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import tkinter as tk
import pynput
import time

class HandTyper:
    
    def __init__(self):
        
        """
        Parameters for the mediapipe tasks
        """
        self.BaseOptions = mp.tasks.BaseOptions
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        """
        Flags for each mode
        """
        self.result_gesture = None
        self.mouse_mode = True
        self.key_mode = False

        """
        dir for the task file (The Model)
        """
        self.model_path = "Hand_Data_V7/gesture_recognizer.task"
        self.options = self.GestureRecognizerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )

        """
        MediaPipe Hands for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def print_result(self, result, output_image: mp.Image, timestamp_ms: int):
        
        """
        Def: Callback function to handle the result of the gesture recognition.
        
        Params: 
        - result: The result of the gesture recognition.
        - output_image: The image with the recognized gestures drawn on it.
        - timestamp_ms: The timestamp of the image in milliseconds.
        """
        
        if result.gestures:
            
            if self.key_mode:
                print(f"Recognized gestures: {result.gestures}") #Only prints out the recognized gesture when key_mode is true
                
            target = str(result.gestures[0][0].category_name)

            if target and target != 'none':
                self.result_gesture = target
                self.translate(self.result_gesture)
            else:
                self.result_gesture = "No gesture detected"

    def draw_menu(self, frame):
        
        """
        Def: Draws the menu on the frame.
        Params: frame - The frame to draw the menu on.
        """
        cv2.rectangle(frame, (0, 0), (320, 50), (200, 200, 200), -1)
        cv2.putText(frame, "Menu: [Q] Quit | [M] Switch Modes (Mouse | Keyboard) ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def main_capture(self):
        
        """
        Def: Main function to capture the video stream and process the hand gestures.
        """
        
        cv2.namedWindow('HandTyperv1', cv2.WINDOW_NORMAL) #Names the Window accordingly for the 
        cv2.resizeWindow('HandTyperv1', self.windowsize()[0], self.windowsize()[1])
        cv2.setWindowProperty('HandTyperv1', cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow('HandTyperv1', (self.windowsize()[0] * 4 - self.windowsize()[0]), (self.windowsize()[1] * 4 - self.windowsize()[1])) # Moves the window to the top left corner of the screen
        

        with self.GestureRecognizer.create_from_options(self.options) as recognizer: # Intialize the recognizer with the predefoined options
            with self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands: # Initialize the MediaPipe Hands module
                cap = cv2.VideoCapture(0) # Open the webcam

                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (self.windowsize()[0], self.windowsize()[1]))
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    frame_timestamp_ms = int(time.time() * 1000)
                    recognizer.recognize_async(mp_image, frame_timestamp_ms)
                    results = hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        if self.mouse_mode:
                            hand_landmarks = results.multi_hand_landmarks[0]
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                            if self.mouse_mode:
                                mouse = pynput.mouse.Controller()

                                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                                index_x = int(index_finger_tip.x * frame.shape[1]) 
                                index_y = int(index_finger_tip.y * frame.shape[0])
                                
                                self.mouse_movement(index_x, index_y) # move the mouse to the index finger tip position
                                
                                middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                                middle_x = int(middle_finger_tip.x * frame.shape[1])
                                middle_y = int(middle_finger_tip.y * frame.shape[0])
                                
                                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1) # draw a circle around the index finger tip
                                
                                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                                thumb_x = int(thumb_tip.x * frame.shape[1])
                                thumb_y = int(thumb_tip.y * frame.shape[0])
                                
                                thumb_index_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5 # calculate the distance between the index finger and thumb
                                mid_index_distance = ((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2) ** 0.5 # calculate the distance between the middle finger and index finger
                                
                                print(f"Thumb distance: {thumb_index_distance}, Middle distance: {mid_index_distance}") # print the distances for debugging
                                
                                if thumb_index_distance < 25: # if the thumb and index finger are close together
                                    mouse.click(pynput.mouse.Button.right, 1) # click the left mouse button    
                                    
                                elif mid_index_distance < 25: # if the middle finger and index finger are close together
                                    mouse.click(pynput.mouse.Button.left, 1) # click the right mouse button  

                    self.draw_menu(frame)
                    cv2.putText(frame, self.result_gesture, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                    cv2.imshow('HandTyperv1', frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    
                    elif key == ord('m'):
                        self.mouse_mode = not self.mouse_mode
                        self.key_mode = not self.key_mode
                        print(f"Mouse mode: {self.mouse_mode}, Key mode: {self.key_mode}")

                cap.release()
                cv2.destroyAllWindows()

    def windowsize(self):
        
        """
        Def: Get the size of the window.
        Returns: A tuple with the width and height of the window.
        """
        main = tk.Tk()
        return main.winfo_screenwidth() // 4, main.winfo_screenheight() // 4


    def mouse_movement(self, x, y):
        
        """
        Function to move the mouse to the x and y coordinates of the index finger tip
        
        Params:
        x: x coordinate of the index finger tip
        y: y coordinate of the index finger tip
        """
        mouse = pynput.mouse.Controller()
        current_x, current_y = mouse.position  # Get the current mouse position


        scaled_x = int(x * int(self.windowsize()[0] * 4) / int(self.windowsize()[1] * 4))
        scaled_y = int(y * int(self.windowsize()[1] * 4) / int(self.windowsize()[0] * 4))# Scale the x coordinate to the screen resolution
        # Number of steps for smooth movement
        steps = 10
        for i in range(1, steps + 1):
            # Interpolate between the current position and the target position
            smooth_x = current_x + (scaled_x - current_x) * i / steps
            smooth_y = current_y + (scaled_y - current_y) * i / steps
            mouse.position = (int(smooth_x), int(smooth_y))
            time.sleep(0.01)  # Add a small delay for smoother movement

    def translate(self, sign):
        
        """
        Def: Translate the recognized gesture into a keyboard or mouse action.
        Oa
        """
        
        keyboard = pynput.keyboard.Controller()
        mouse = pynput.mouse.Controller()
        target = sign.split()

        if self.mouse_mode:
            
            if sign == "CLICK":
                mouse.click(pynput.mouse.Button.left, 1)
            
            elif sign == "RIGHT_CLICK":
                mouse.click(pynput.mouse.Button.right, 1) 
        
        else: 
            if target == 'BACKSPACE':
                keyboard.press(pynput.keyboard.Key.backspace)
                keyboard.release(pynput.keyboard.Key.backspace)
                
            elif target == 'ENTER':
                keyboard.press(pynput.keyboard.Key.space)
                keyboard.release(pynput.keyboard.Key.space)
                
            elif target == 'SPACE':
                keyboard.press(pynput.keyboard.Key.enter)
                keyboard.release(pynput.keyboard.Key.enter)

                
            elif target == '10':
                keyboard.press(pynput.keyboard.KeyCode.from_char('1'))
                keyboard.release(pynput.keyboard.KeyCode.from_char('1'))
                keyboard.press(pynput.keyboard.KeyCode.from_char('0'))
                keyboard.release(pynput.keyboard.KeyCode.from_char('0'))
                
            else:
                keyboard.press(target[-1])
            print(f"Typed: {target[-1]}")


if __name__ == '__main__':
    hand_typer = HandTyper()
    hand_typer.main_capture()
