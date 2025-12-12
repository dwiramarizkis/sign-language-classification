# Deploy ASL Detection ke Railway

## Persiapan

1. **Rename requirements file:**
   ```bash
   mv requirements_flask.txt requirements.txt
   ```

2. **Pastikan file-file ini ada:**
   - `flask_app.py` - Main Flask application
   - `templates/index.html` - Frontend HTML
   - `requirements.txt` - Python dependencies
   - `Procfile` - Railway start command
   - `railway.json` - Railway configuration
   - `asl_model.h5` - Trained model
   - `label_encoder.pkl` - Label encoder
   - `runtime.txt` - Python version (3.11)

## Deploy ke Railway

### Opsi 1: Via Railway Dashboard

1. Buka https://railway.app/
2. Login dengan GitHub
3. Klik **"New Project"**
4. Pilih **"Deploy from GitHub repo"**
5. Pilih repository: `dwiramarizkis/sign-language-classification`
6. Railway akan otomatis detect dan deploy
7. Tunggu build selesai
8. Klik **"Generate Domain"** untuk mendapatkan URL publik

### Opsi 2: Via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up
```

## Environment Variables (Opsional)

Tidak ada environment variables yang diperlukan untuk project ini.

## Verifikasi Deployment

1. Buka URL yang diberikan Railway
2. Klik "Start Camera"
3. Izinkan akses kamera
4. Tunjukkan gesture ASL
5. Lihat prediksi real-time

## Troubleshooting

### Build Failed
- Pastikan `requirements.txt` ada dan benar
- Pastikan Python 3.11 di `runtime.txt`

### Model Not Found
- Pastikan `asl_model.h5` dan `label_encoder.pkl` ada di repository
- Check `.gitignore` tidak mengexclude file model

### Camera Not Working
- Pastikan menggunakan HTTPS (Railway otomatis provide)
- Browser modern diperlukan (Chrome, Firefox, Edge)
- Izinkan akses kamera di browser

## Fitur

✅ Real-time detection via webcam
✅ Automatic prediction setiap 500ms
✅ Flip horizontal otomatis
✅ Responsive design
✅ No external dependencies (self-contained)

## Tech Stack

- **Backend:** Flask + Gunicorn
- **ML:** TensorFlow + MediaPipe
- **Frontend:** Vanilla JavaScript + HTML5 Canvas
- **Hosting:** Railway

## Estimasi Resource

- **Memory:** ~500MB
- **Build Time:** 3-5 menit
- **Cold Start:** 5-10 detik
