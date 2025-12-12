"""
ASL Sign Language Detection - Real-time Detection
Hybrid Approach: Landmarks + Extra Features
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow import keras
from collections import deque

# ============================================================================
# 2. DATA UNDERSTANDING (Load Model & Encoder)
# ============================================================================
print("="*70)
print("ASL SIGN LANGUAGE DETECTION - REAL-TIME")
print("="*70)
print("Loading model dan label encoder...")

# Load model
model_filename = "asl_model.h5"
try:
    model = keras.models.load_model(model_filename)
    print(f"✓ Model berhasil dimuat: {model_filename}")
except:
    print(f"ERROR: File {model_filename} tidak ditemukan!")
    print("Jalankan train.py terlebih dahulu.")
    exit()

# Load label encoder
encoder_filename = "label_encoder.pkl"
try:
    with open(encoder_filename, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder berhasil dimuat: {encoder_filename}")
except:
    print(f"ERROR: File {encoder_filename} tidak ditemukan!")
    print("Jalankan train.py terlebih dahulu.")
    exit()

print(f"✓ Jumlah kelas: {len(label_encoder.classes_)}")
print(f"✓ Kelas: {label_encoder.classes_}")
print()

# ============================================================================
# 3. DATA PREPARATION (Setup MediaPipe & Camera)
# ============================================================================
print("Menginisialisasi MediaPipe Hands dan Webcam...")

# Inisialisasi MediaPipe Hands dengan konfigurasi optimal untuk real-time
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Tidak dapat mengakses webcam!")
    exit()

# Set resolusi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✓ Webcam berhasil diinisialisasi")
print()
print("="*70)
print("DETEKSI DIMULAI")
print("="*70)
print("Instruksi:")
print("  - Tunjukkan tangan Anda ke kamera")
print("  - Tekan 'q' untuk keluar")
print("  - Tekan 'c' untuk clear prediction history")
print("="*70)
print()

# ============================================================================
# 4. MODELING & 5. EVALUATION (Real-time Prediction Loop)
# ============================================================================

# Variabel untuk smoothing prediksi (mengurangi flickering)
prediction_history = deque(maxlen=5)  # Simpan 5 prediksi terakhir
confidence_threshold = 0.7

# Counter
frame_count = 0

# Loop utama
while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Gagal membaca frame dari webcam")
        break
    
    frame_count += 1
    
    # Flip frame horizontal (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # Get dimensi frame
    h, w, c = frame.shape
    
    # Convert BGR to RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi tangan
    results = hands.process(frame_rgb)
    
    # Variabel untuk prediksi
    prediction_text = ""
    confidence_score = 0.0
    
    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        # Ambil landmark tangan pertama
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Gambar landmark di frame dengan style yang lebih baik
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Ekstrak koordinat x, y dari 21 landmark
        x_coords = []
        y_coords = []
        
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalisasi koordinat
        min_x = min(x_coords)
        min_y = min(y_coords)
        
        data_aux = []
        
        # Tambahkan koordinat yang dinormalisasi (42 features)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)
        
        # FITUR TAMBAHAN (3 features) - SAMA seperti training
        
        # 1. Palm orientation
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        palm_angle = np.arctan2(middle_mcp.y - wrist.y, 
                               middle_mcp.x - wrist.x)
        data_aux.append(palm_angle)
        
        # 2. Hand openness
        thumb_tip = hand_landmarks.landmark[4]
        pinky_tip = hand_landmarks.landmark[20]
        openness = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + 
                          (thumb_tip.y - pinky_tip.y)**2)
        data_aux.append(openness)
        
        # 3. Finger curl
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
        
        # Convert ke numpy array (total 45 features)
        landmarks_array = np.array(data_aux, dtype=np.float32).reshape(1, -1)
        
        # Prediksi menggunakan model
        predictions = model.predict(landmarks_array, verbose=0)
        
        # Ambil kelas dengan probabilitas tertinggi
        predicted_class_idx = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_idx]
        
        # Decode label
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Tambahkan ke history untuk smoothing
        if confidence_score > confidence_threshold:
            prediction_history.append(predicted_label)
        
        # Ambil prediksi yang paling sering muncul (majority voting)
        if len(prediction_history) > 0:
            # Count occurrences
            from collections import Counter
            counter = Counter(prediction_history)
            prediction_text = counter.most_common(1)[0][0]
        
        # Tampilkan prediksi di frame
        if confidence_score > confidence_threshold and prediction_text:
            # Background box untuk teks
            box_height = 120
            cv2.rectangle(frame, (10, 10), (350, box_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, box_height), (0, 255, 0), 2)
            
            # Teks prediksi (huruf besar)
            cv2.putText(
                frame,
                f"Huruf: {prediction_text.upper()}",
                (25, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )
            
            # Teks confidence
            cv2.putText(
                frame,
                f"Confidence: {confidence_score*100:.1f}%",
                (25, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            # Confidence rendah
            cv2.putText(
                frame,
                "Confidence rendah - Posisikan tangan lebih jelas",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )
    else:
        # Tidak ada tangan terdeteksi
        cv2.putText(
            frame,
            "Tidak ada tangan terdeteksi",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        
        # Clear prediction history jika tidak ada tangan
        prediction_history.clear()
    
    # Tampilkan instruksi di bagian bawah
    cv2.putText(
        frame,
        "Tekan 'q' untuk keluar | 'c' untuk clear history",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Tampilkan FPS
    if frame_count > 10:  # Skip beberapa frame pertama
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (w - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    # Tampilkan frame
    cv2.imshow('ASL Sign Language Detection - Real-time', frame)
    
    # Cek keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\nDeteksi dihentikan oleh user.")
        break
    elif key == ord('c') or key == ord('C'):
        prediction_history.clear()
        print("Prediction history cleared.")

# ============================================================================
# 6. DEPLOYMENT (Cleanup)
# ============================================================================
print("\nMembersihkan resources...")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print("✓ Webcam dilepas")
print("✓ Window ditutup")
print()
print("="*70)
print("PROGRAM SELESAI")
print("="*70)
