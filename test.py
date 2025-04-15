import cv2

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if not ret:
        print("Failed to capture frame. Exiting...")
        break    
    
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the loop
        break