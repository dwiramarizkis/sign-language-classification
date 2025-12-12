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
    import os
    try:
        # Debug: Check current directory and files
        current_dir = os.getcwd()
        files_in_dir = os.listdir(current_dir)
        
        model_path = os.path.join(current_dir, 'asl_model.h5')
        encoder_path = os.path.join(current_dir, 'label_encoder.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            return None, None, f"Model file not found. Current dir: {current_dir}, Files: {files_in_dir[:10]}"
        
        if not os.path.exists(encoder_path):
            return None, None, f"Encoder file not found. Current dir: {current_dir}, Files: {files_in_dir[:10]}"
        
        model = keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder, None
    except Exception as e:
        return None, None, f"Error: {str(e)}"

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
    **Opsi 1: Gunakan Camera**
    1. Pilih tab **Camera**
    2. Klik tombol kamera
    3. Izinkan akses kamera
    4. Posisikan gesture ASL
    5. Ambil foto
    
    **Opsi 2: Upload File**
    1. Pilih tab **Upload File**
    2. Klik **Browse files**
    3. Pilih foto gesture ASL
    4. Lihat hasil prediksi
    """)
    
    # Main content
    st.markdown("### 📸 Input Gambar")
    
    # Tabs untuk camera dan upload
    tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload File"])
    
    camera_photo = None
    uploaded_file = None
    
    with tab1:
        st.info("💡 **Cara Penggunaan:** Klik tombol kamera, izinkan akses, ambil foto gesture ASL Anda.")
        camera_photo = st.camera_input("Ambil foto tangan Anda")
    
    with tab2:
        st.info("💡 **Cara Penggunaan:** Upload foto tangan dengan gesture ASL (format: JPG, JPEG, PNG).")
        uploaded_file = st.file_uploader("Upload foto tangan Anda", type=['jpg', 'jpeg', 'png'])
    
    # Gunakan input yang tersedia
    input_image = camera_photo if camera_photo is not None else uploaded_file
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_image is not None:
            # Read image
            file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Flip horizontal untuk mirror effect (hanya untuk camera input)
            if camera_photo is not None:
                frame = cv2.flip(frame, 1)
            
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
                
                if confidence_score > confidence_threshold:
                    prediction_text = predicted_label
                
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
            
            # Tampilkan hasil
            st.image(frame_rgb, channels="RGB", use_column_width=True, caption="Hasil Deteksi")
            
            # Store in session state for col2
            st.session_state.prediction_text = prediction_text
            st.session_state.confidence_score = confidence_score
            st.session_state.hand_detected = hand_detected
        else:
            st.info("📷 Gunakan tab **Camera** untuk mengambil foto atau tab **Upload File** untuk upload gambar")
            st.session_state.prediction_text = ""
            st.session_state.confidence_score = 0.0
            st.session_state.hand_detected = False
    
    with col2:
        st.markdown("### 🎯 Prediksi")
        
        if hasattr(st.session_state, 'prediction_text') and st.session_state.prediction_text and st.session_state.confidence_score > confidence_threshold:
            st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-letter">{st.session_state.prediction_text.upper()}</div>
                    <div class="confidence-text">Confidence: {st.session_state.confidence_score*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-letter">-</div>
                    <div class="confidence-text">Menunggu foto...</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Status")
        if hasattr(st.session_state, 'hand_detected'):
            st.markdown(f"""
                <div class="stats-box">
                    <strong>Hand Detected:</strong> {'✅ Yes' if st.session_state.hand_detected else '❌ No'}<br>
                    <strong>Status:</strong> {'✅ Ready' if input_image else '⏳ Waiting'}
                </div>
            """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
