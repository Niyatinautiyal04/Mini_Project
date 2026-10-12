from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load your trained Scikit-Learn model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Configure MediaPipe Tasks API
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7)

landmarker = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

current_prediction = ""

def generate_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)
        
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            h, w, _ = frame.shape
            
            # Extract features for machine learning
            features = []
            
            # Get the wrist coordinates to act as our (0,0) anchor
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            
            for lm in hand_landmarks:
                # Subtract wrist position to normalize the data
                features.extend([lm.x - wrist_x, lm.y - wrist_y])
            
            # Predict
            prediction = model.predict(np.array(features).reshape(1, -1))[0]
            current_prediction = str(prediction)
            
            # Draw skeleton
            for connection in HAND_CONNECTIONS:
                p1, p2 = hand_landmarks[connection[0]], hand_landmarks[connection[1]]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
            
            cv2.putText(frame, f'Sign: {prediction}', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sign')
def get_sign():
    return jsonify({'sign': current_prediction})

if __name__ == "__main__":
    app.run(debug=True)