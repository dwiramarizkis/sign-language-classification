# 🤟 ASL Sign Language Detection

Sistem deteksi bahasa isyarat ASL (American Sign Language) real-time menggunakan Deep Learning dan MediaPipe Hands.

## 📋 Fitur

- ✅ Deteksi real-time via webcam
- ✅ 26 huruf ASL (A-Z)
- ✅ Akurasi tinggi dengan 45 features (landmarks + extra features)
- ✅ Web interface dengan Streamlit
- ✅ Prediction smoothing untuk hasil stabil
- ✅ Confidence threshold yang dapat disesuaikan

## 🚀 Instalasi

### 1. Clone atau Download Repository

```bash
git clone <repository-url>
cd asl-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Dataset

Pastikan Anda memiliki dataset dengan struktur folder:

```
asl_dataset/
├── a/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── b/
│   ├── image1.jpg
│   └── ...
├── c/
└── ...
```

## 🎯 Cara Menggunakan

### 1. Training Model

Jalankan script training untuk melatih model:

```bash
python train.py
```

Output:
- `asl_model.h5` - Model terlatih
- `label_encoder.pkl` - Label encoder
- `training_history.png` - Grafik training

### 2. Testing Real-time (OpenCV)

Untuk testing dengan OpenCV (tanpa web interface):

```bash
python realtime.py
```

Kontrol:
- `q` - Keluar
- `c` - Clear prediction history

### 3. Web App (Streamlit)

Untuk menjalankan web interface:

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser (default: http://localhost:8501)

## 🎨 Fitur Web App

### Sidebar
- **Confidence Threshold**: Atur minimum confidence untuk prediksi
- **Smoothing Window**: Atur jumlah frame untuk smoothing
- **Tampilkan Landmarks**: Toggle visualisasi hand landmarks
- **Tampilkan Confidence**: Toggle tampilan confidence score

### Main Interface
- **Video Stream**: Real-time video dari webcam
- **Prediksi**: Huruf yang terdeteksi dengan confidence score
- **Statistik**: Frame count, FPS, detection rate

## 🧠 Teknologi

- **MediaPipe Hands**: Ekstraksi 21 hand landmarks
- **TensorFlow/Keras**: Deep Neural Network (1024→512→256→128→64→output)
- **OpenCV**: Video capture dan image processing
- **Streamlit**: Web interface
- **Scikit-learn**: Data preprocessing dan evaluation

## 📊 Arsitektur Model

```
Input (45 features)
    ↓
Dense(1024) + BatchNorm + Dropout(0.4)
    ↓
Dense(512) + BatchNorm + Dropout(0.4)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + Dropout(0.2)
    ↓
Output (26 classes - Softmax)
```

## 🔍 Features Extraction

**42 Basic Features:**
- 21 hand landmarks × 2 (x, y coordinates)
- Normalized relative to bounding box

**3 Extra Features:**
1. **Palm Orientation**: Sudut antara wrist dan middle finger MCP
2. **Hand Openness**: Jarak antara thumb tip dan pinky tip
3. **Finger Curl**: Rata-rata jarak fingertips ke pusat palm

Total: **45 features**

## 📈 Training Details

- **Optimizer**: Adam (learning rate: 0.0005)
- **Loss**: Sparse Categorical Crossentropy
- **Callbacks**:
  - EarlyStopping (patience: 20)
  - ReduceLROnPlateau (patience: 7)
  - ModelCheckpoint (save best model)
- **Class Weighting**: Balanced untuk imbalanced data
- **Epochs**: 150 (dengan early stopping)
- **Batch Size**: 32

## 🎯 Tips untuk Akurasi Maksimal

1. **Pencahayaan**: Gunakan pencahayaan yang baik dan merata
2. **Background**: Gunakan background yang kontras dengan warna kulit
3. **Jarak**: Posisikan tangan pada jarak 30-50cm dari kamera
4. **Posisi**: Pastikan seluruh tangan terlihat di frame
5. **Gesture**: Tunjukkan gesture dengan jelas dan stabil

## 🐛 Troubleshooting

### Model tidak ditemukan
```
ERROR: File asl_model.h5 tidak ditemukan!
```
**Solusi**: Jalankan `python train.py` terlebih dahulu

### Webcam tidak terdeteksi
```
ERROR: Tidak dapat mengakses webcam!
```
**Solusi**: 
- Pastikan webcam terhubung
- Cek permission kamera di sistem operasi
- Coba ganti index kamera: `cv2.VideoCapture(1)` atau `cv2.VideoCapture(2)`

### Akurasi rendah
**Solusi**:
- Tambah data training
- Sesuaikan confidence threshold
- Tingkatkan smoothing window
- Pastikan pencahayaan baik

## 📝 File Structure

```
asl-detection/
├── asl_dataset/          # Dataset gambar
├── train.py              # Script training
├── realtime.py           # Real-time detection (OpenCV)
├── app.py                # Web app (Streamlit)
├── requirements.txt      # Dependencies
├── README.md             # Dokumentasi
├── asl_model.h5          # Model terlatih (generated)
├── label_encoder.pkl     # Label encoder (generated)
└── training_history.png  # Grafik training (generated)
```

## 📄 License

MIT License

## 👨‍💻 Author

Senior Computer Vision Engineer

## 🙏 Acknowledgments

- MediaPipe by Google
- TensorFlow/Keras
- Streamlit
- OpenCV
