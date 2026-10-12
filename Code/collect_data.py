import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import urllib.request

# 1. Download the required Tasks API model if missing
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading Google's hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# 2. Configure MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5)

landmarker = HandLandmarker.create_from_options(options)

# Manually define connections since drawing_utils was removed
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

# 3. Setup CSV
csv_file = 'sign_mnist.csv'
if not os.path.exists(csv_file):
    columns = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

cap = cv2.VideoCapture(0)
current_label = input("Enter the label (e.g., 'A', 'B', 'C') for the sign you are about to show: ")
print("Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Tasks API requires an mp.Image object
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Process the image
    result = landmarker.detect(mp_image)
    
    if result.hand_landmarks:
        # Get the first hand detected
        hand_landmarks = result.hand_landmarks[0]
        h, w, _ = frame.shape
        
        # Draw connections manually
        for connection in HAND_CONNECTIONS:
            p1, p2 = hand_landmarks[connection[0]], hand_landmarks[connection[1]]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw joint points
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            # Save data on 's' press
        if cv2.waitKey(1) & 0xFF == ord('s'):
            row = [current_label]
            
            # Get the wrist coordinates to act as our (0,0) anchor
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            
            for lm in hand_landmarks:
                # Subtract wrist position to normalize the data
                row.extend([lm.x - wrist_x, lm.y - wrist_y])
                
            pd.DataFrame([row]).to_csv(csv_file, mode='a', header=False, index=False)
            print(f"Saved 1 sample for {current_label}")

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()