"""
ASL Sign Language Detection - Training Script
Hybrid Approach: Landmarks + Extra Features untuk Akurasi Tinggi
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ============================================================================
# 2. DATA UNDERSTANDING
# ============================================================================
print("="*70)
print("2. DATA UNDERSTANDING")
print("="*70)

DATA_DIR = "asl_dataset"

# Cek keberadaan dataset
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Folder {DATA_DIR} tidak ditemukan!")
    exit()

# List semua kelas
classes = sorted([d for d in os.listdir(DATA_DIR) 
                  if os.path.isdir(os.path.join(DATA_DIR, d))])

print(f"Jumlah kelas: {len(classes)}")
print(f"Kelas: {classes}")

# Hitung total gambar
total_images = 0
for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    num_images = len([f for f in os.listdir(cls_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_images += num_images
    if cls in classes[:5]:  # Tampilkan 5 kelas pertama
        print(f"  - Kelas '{cls}': {num_images} gambar")

print(f"\nTotal gambar dalam dataset: {total_images}")
print()

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("="*70)
print("3. DATA PREPARATION")
print("="*70)
print("Ekstraksi landmark dengan MediaPipe Hands...")

# Inisialisasi MediaPipe dengan multiple configurations untuk deteksi lebih robust
mp_hands = mp.solutions.hands

# Gunakan beberapa konfigurasi untuk meningkatkan deteksi
hands_configs = [
    mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, 
                   max_num_hands=1, model_complexity=1),
    mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, 
                   max_num_hands=1, model_complexity=1),
    mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.05, 
                   max_num_hands=1, model_complexity=0),
]

# Storage untuk data
X_data = []
y_data = []

# Counter
total_processed = 0
total_failed = 0

# Loop setiap kelas
for class_name in classes:
    class_path = os.path.join(DATA_DIR, class_name)
    
    # Ambil semua file gambar
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Memproses kelas '{class_name}': {len(image_files)} gambar...")
    
    class_success = 0
    
    # Loop setiap gambar
    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)
        
        # Baca gambar
        img = cv2.imread(img_path)
        
        if img is None:
            total_failed += 1
            continue
        
        # Coba deteksi dengan multiple configurations
        detected = False
        
        for hands in hands_configs:
            # Resize jika gambar terlalu kecil
            h, w = img.shape[:2]
            if h < 100 or w < 100:
                scale = max(100 / h, 100 / w)
                img_test = cv2.resize(img, None, fx=scale, fy=scale)
            else:
                img_test = img
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
            
            # Deteksi landmark
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Ekstrak koordinat x, y dari 21 landmark
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                
                # Normalisasi koordinat relatif terhadap bounding box
                min_x = min(x_coords)
                min_y = min(y_coords)
                
                data_aux = []
                
                # Tambahkan koordinat yang dinormalisasi (21 landmark * 2 = 42 features)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)
                
                # FITUR TAMBAHAN untuk meningkatkan akurasi (3 features)
                
                # 1. Palm orientation (sudut antara wrist dan middle finger MCP)
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                palm_angle = np.arctan2(middle_mcp.y - wrist.y, 
                                       middle_mcp.x - wrist.x)
                data_aux.append(palm_angle)
                
                # 2. Hand openness (jarak antara thumb tip dan pinky tip)
                thumb_tip = hand_landmarks.landmark[4]
                pinky_tip = hand_landmarks.landmark[20]
                openness = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + 
                                  (thumb_tip.y - pinky_tip.y)**2)
                data_aux.append(openness)
                
                # 3. Finger curl (rata-rata jarak fingertips ke pusat palm)
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
                
                # Total features: 42 + 3 = 45
                X_data.append(data_aux)
                y_data.append(class_name)
                
                total_processed += 1
                class_success += 1
                detected = True
                break
        
        if not detected:
            total_failed += 1
    
    print(f"  Berhasil: {class_success}/{len(image_files)}")

# Tutup semua MediaPipe instances
for hands in hands_configs:
    hands.close()

print()
print(f"Total berhasil diproses: {total_processed}")
print(f"Total gagal/tidak terdeteksi: {total_failed}")
print()

# Convert ke numpy array
X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data)

print(f"Shape data X: {X_data.shape}")  # (samples, 45)
print(f"Shape data y: {y_data.shape}")
print()

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_data)

num_classes = len(label_encoder.classes_)
print(f"Jumlah kelas: {num_classes}")
print(f"Label classes: {label_encoder.classes_}")
print()

# Cek distribusi kelas
print("Distribusi sampel per kelas:")
unique, counts = np.unique(y_encoded, return_counts=True)
for idx, count in zip(unique, counts):
    class_name = label_encoder.inverse_transform([idx])[0]
    print(f"  - Kelas '{class_name}': {count} sampel")
print()

# Filter kelas dengan sampel < 2
min_samples = 2
valid_indices = []
for i, label in enumerate(y_encoded):
    if counts[label] >= min_samples:
        valid_indices.append(i)

if len(valid_indices) < len(y_encoded):
    print(f"⚠ WARNING: {len(y_encoded) - len(valid_indices)} sampel dibuang (kelas < {min_samples} sampel)")
    X_data = X_data[valid_indices]
    y_encoded = y_encoded[valid_indices]
    print(f"Data setelah filtering: {len(X_data)} sampel\n")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_encoded, 
    test_size=0.2, 
    shuffle=True, 
    stratify=y_encoded, 
    random_state=42
)

print(f"Data training: {X_train.shape[0]} sampel")
print(f"Data testing: {X_test.shape[0]} sampel")
print()

# ============================================================================
# 4. MODELING
# ============================================================================
print("="*70)
print("4. MODELING")
print("="*70)
print("Membuat arsitektur Deep Neural Network...")

# Input shape
input_shape = X_train.shape[1]  # 45 features

# Arsitektur model yang lebih dalam untuk akurasi tinggi
model = keras.Sequential([
    layers.Input(shape=(input_shape,)),
    
    # Layer 1: Deep feature extraction
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # Layer 2
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # Layer 3
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Layer 4
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Layer 5
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Tampilkan summary
model.summary()
print()

# ============================================================================
# 5. EVALUATION
# ============================================================================
print("="*70)
print("5. EVALUATION")
print("="*70)
print("Melatih model dengan callbacks...")

# Compute class weights untuk menangani imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Callbacks untuk training yang lebih baik
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'asl_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print()

# Evaluasi final
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print()

# ============================================================================
# 6. DEPLOYMENT
# ============================================================================
print("="*70)
print("6. DEPLOYMENT")
print("="*70)
print("Menyimpan model dan label encoder...")

# Load best model dan simpan sebagai model final
best_model = keras.models.load_model('asl_model_best.h5')
best_model.save('asl_model.h5')
print(f"✓ Model disimpan: asl_model.h5")

# Simpan label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ Label encoder disimpan: label_encoder.pkl")

print()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print(f"✓ Grafik training disimpan: training_history.png")

print()
print("="*70)
print("TRAINING SELESAI!")
print("="*70)
print(f"Akurasi Final: {test_accuracy*100:.2f}%")
print(f"Jalankan: python realtime.py")
print("="*70)
