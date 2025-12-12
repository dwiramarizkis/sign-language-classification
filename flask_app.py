"""
ASL Sign Language Detection - Flask Web App
Deploy to Railway
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow import keras
import base64
import os

app = Flask(__name__)

# ============================================================================
# LOAD MODEL & ENCODER
# ============================================================================
print("Loading model...")
model = keras.models.load_model('asl_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Model loaded successfully!")

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=1
)

# ============================================================================
# EXTRACT FEATURES
# ============================================================================
def extract_features(hand_landmarks):
    """Ekstrak 45 features dari hand landmarks"""
    x_coords = []
    y_coords = []
    
    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)
    
    min_x = min(x_coords)
    min_y = min(y_coords)
    
    data_aux = []
    
    # 42 features: koordinat yang dinormalisasi
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min_x)
        data_aux.append(y - min_y)
    
    # 3 features tambahan
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    palm_angle = np.arctan2(middle_mcp.y - wrist.y, 
                           middle_mcp.x - wrist.x)
    data_aux.append(palm_angle)
    
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    openness = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + 
                      (thumb_tip.y - pinky_tip.y)**2)
    data_aux.append(openness)
    
    palm_center_x = np.mean([hand_landmarks.landmark[i].x 
                            for i in [0, 5, 9, 13, 17]])
    palm_center_y = np.mean([hand_landmarks.landmark[i].y 
                            for i in [0, 5, 9, 13, 17]])
    fingertips = [4, 8, 12, 16, 20]
    avg_curl = np.mean([
        np.sqrt((hand_landmarks.landmark[i].x - palm_center_x)**2 + 
               (hand_landmarks.landmark[i].y - palm_center_y)**2)
        for i in fingertips
    ])
    data_aux.append(avg_curl)
    
    return np.array(data_aux, dtype=np.float32).reshape(1, -1)

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Flip horizontal
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract features
            features = extract_features(hand_landmarks)
            
            # Predict
            predictions = model.predict(features, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence_score = float(predictions[0][predicted_class_idx])
            predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return jsonify({
                'success': True,
                'prediction': predicted_label.upper(),
                'confidence': round(confidence_score * 100, 1),
                'hand_detected': True
            })
        else:
            return jsonify({
                'success': True,
                'prediction': None,
                'confidence': 0,
                'hand_detected': False,
                'message': 'Tidak ada tangan terdeteksi'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
