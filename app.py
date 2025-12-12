"""
ASL Sign Language Detection - Streamlit Web App
Real-time detection via webcam dengan interface yang user-friendly
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow import keras
from collections import deque
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="ASL Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #1E88E5;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: bold;
        color: #1E88E5;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stats-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & ENCODER
# ============================================================================
@st.cache_resource
def load_model_and_encoder():
    """Load model dan label encoder (cached)"""
    try:
        model = keras.models.load_model('asl_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder, None
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================
@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe Hands (cached)"""
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
    
    return hands, mp_hands, mp_drawing, mp_drawing_styles

# ============================================================================
# EXTRACT FEATURES FROM LANDMARKS
# ============================================================================
def extract_features(hand_landmarks):
    """Ekstrak 45 features dari hand landmarks"""
    # Ekstrak koordinat x, y
    x_coords = []
    y_coords = []
    
    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)
    
    # Normalisasi
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
    
    return np.array(data_aux, dtype=np.float32).reshape(1, -1)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">🤟 ASL Sign Language Detection</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time American Sign Language Recognition</div>', 
                unsafe_allow_html=True)
    
    # Load model
    model, label_encoder, error = load_model_and_encoder()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Pastikan Anda sudah menjalankan `train.py` terlebih dahulu!")
        st.stop()
    
    st.success("✅ Model berhasil dimuat!")
    
    # Initialize MediaPipe
    hands, mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
    
    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence untuk menampilkan prediksi"
    )
    
    # Smoothing window
    smoothing_window = st.sidebar.slider(
        "Smoothing Window",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Jumlah frame untuk smoothing prediksi"
    )
    
    # Show landmarks
    show_landmarks = st.sidebar.checkbox("Tampilkan Landmarks", value=True)
    
    # Show confidence
    show_confidence = st.sidebar.checkbox("Tampilkan Confidence", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Informasi Model")
    st.sidebar.info(f"**Jumlah Kelas:** {len(label_encoder.classes_)}\n\n**Kelas:** {', '.join(label_encoder.classes_)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Instruksi")
    st.sidebar.markdown("""
    1. Klik tombol **Start Detection**
    2. Izinkan akses kamera
    3. Tunjukkan tangan Anda
    4. Lihat prediksi real-time
    5. Klik **Stop Detection** untuk berhenti
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Video Stream")
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🎯 Prediksi")
        prediction_placeholder = st.empty()
        
        st.markdown("### 📈 Statistik")
        stats_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        start_button = st.button("▶️ Start Detection", use_container_width=True)
    
    with col_btn2:
        stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
    
    # Session state untuk kontrol
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    
    if start_button:
        st.session_state.detection_running = True
    
    if stop_button:
        st.session_state.detection_running = False
    
    # Detection loop
    if st.session_state.detection_running:
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Tidak dapat mengakses webcam!")
            st.session_state.detection_running = False
            st.stop()
        
        # Set resolusi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Prediction history untuk smoothing
        prediction_history = deque(maxlen=smoothing_window)
        
        # Stats
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        
        # Loop
        while st.session_state.detection_running:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Gagal membaca frame dari webcam")
                break
            
            frame_count += 1
            
            # Flip horizontal
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Deteksi tangan
            results = hands.process(frame_rgb)
            
            # Variabel prediksi
            prediction_text = ""
            confidence_score = 0.0
            hand_detected = False
            
            # Jika tangan terdeteksi
            if results.multi_hand_landmarks:
                hand_detected = True
                detection_count += 1
                
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Gambar landmarks
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Ekstrak features
                features = extract_features(hand_landmarks)
                
                # Prediksi
                predictions = model.predict(features, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence_score = predictions[0][predicted_class_idx]
                predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
                
                # Smoothing
                if confidence_score > confidence_threshold:
                    prediction_history.append(predicted_label)
                
                # Majority voting
                if len(prediction_history) > 0:
                    from collections import Counter
                    counter = Counter(prediction_history)
                    prediction_text = counter.most_common(1)[0][0]
                
                # Tampilkan di frame
                if confidence_score > confidence_threshold and prediction_text:
                    # Background box
                    cv2.rectangle(frame_rgb, (10, 10), (300, 100), (30, 144, 255), -1)
                    cv2.rectangle(frame_rgb, (10, 10), (300, 100), (0, 255, 0), 3)
                    
                    # Text
                    cv2.putText(
                        frame_rgb,
                        f"Huruf: {prediction_text.upper()}",
                        (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3
                    )
                    
                    if show_confidence:
                        cv2.putText(
                            frame_rgb,
                            f"Conf: {confidence_score*100:.1f}%",
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
            else:
                # Clear history jika tidak ada tangan
                prediction_history.clear()
                
                # Tampilkan pesan
                cv2.putText(
                    frame_rgb,
                    "Tidak ada tangan terdeteksi",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )
            
            # Tampilkan frame di Streamlit
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update prediction box
            if prediction_text and confidence_score > confidence_threshold:
                prediction_placeholder.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-letter">{prediction_text.upper()}</div>
                        <div class="confidence-text">Confidence: {confidence_score*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                prediction_placeholder.markdown("""
                    <div class="prediction-box">
                        <div class="prediction-letter">-</div>
                        <div class="confidence-text">Menunggu deteksi...</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Update stats
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
            
            stats_placeholder.markdown(f"""
                <div class="stats-box">
                    <strong>Frame:</strong> {frame_count}<br>
                    <strong>FPS:</strong> {fps:.1f}<br>
                    <strong>Detection Rate:</strong> {detection_rate:.1f}%<br>
                    <strong>Hand Detected:</strong> {'✅ Yes' if hand_detected else '❌ No'}
                </div>
            """, unsafe_allow_html=True)
            
            # Small delay
            time.sleep(0.01)
        
        # Release camera
        cap.release()
        st.info("✅ Detection stopped")
    
    else:
        # Placeholder ketika tidak running
        video_placeholder.info("👆 Klik **Start Detection** untuk memulai")
        prediction_placeholder.markdown("""
            <div class="prediction-box">
                <div class="prediction-letter">-</div>
                <div class="confidence-text">Belum dimulai</div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
