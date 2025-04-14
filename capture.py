import cv2
import os
import time

cap = cv2.VideoCapture(0)
# Set the resolution of the camera
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Ensure the output directory exists
output_dir = 'D:\\OneDrive - University of Greenwich\\Modules\\Final Year Project\\HandTyper_FINAL\\SPACE'
os.makedirs(output_dir, exist_ok=True)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('z'):  # Start capturing frames
        time.sleep(3)
        for i in range(4000, 5000):
            # Capture a new frame inside the loop
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting loop...")
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            if i % 5 == 0:  # Save every 5th frame
                cv2.imwrite(os.path.join(output_dir, f'{i}.png'), frame)
                print(f"Captured frame {i}")
                
    elif key == ord('q'):  # Quit the loop
        break

cap.release()
cv2.destroyAllWindows()
