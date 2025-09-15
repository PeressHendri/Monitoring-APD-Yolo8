# APD Monitoring System

Sistem monitoring Alat Pelindung Diri (APD) berbasis YOLO v8 dengan web interface real-time.

## 🚀 Fitur

- **Real-time Detection**: Deteksi helm dan rompi keselamatan secara real-time
- **Web Interface**: Monitoring melalui browser web dengan streaming video
- **Alarm System**: Notifikasi suara otomatis saat pelanggaran
- **Violation Tracking**: Tracking dan logging pelanggaran per person
- **Database Storage**: Penyimpanan riwayat pelanggaran di SQLite
- **Face Detection**: Deteksi wajah untuk tracking person

## 📋 Class Detection

- `Helmet` - Pakai Helm
- `No_Helmet` - Tidak Pakai Helm
- `Vest` - Pakai Rompi
- `No_Vest` - Tidak Pakai Rompi

## 🛠️ Instalasi

### Prerequisites
```bash
pip install ultralytics opencv-python flask pygame numpy
```

### Setup
1. Clone repository
2. Download model weights (jika belum ada)
3. Jalankan sistem:
```bash
python web_monitor.py
```

## 🎯 Penggunaan

### Web Monitor
```bash
python web_monitor.py
```
Akses: http://localhost:5000

### CCTV Monitor
```bash
python cctv_monitor.py
```

### Run All Systems
```bash
python run_all_systems.py
```

## 📁 Struktur Project

```
TESTYOLO8/
├── web_monitor.py          # Web interface utama
├── cctv_monitor.py         # CCTV monitoring
├── run_all_systems.py      # Script untuk menjalankan semua sistem
├── templates/              # HTML templates
├── Combined-APD/           # Dataset training
├── runs/train/             # Model weights hasil training
├── violation_images/       # Gambar pelanggaran yang tersimpan
├── alarm.wav              # File suara alarm
└── apd_violations.db      # Database SQLite
```

## 🔧 Konfigurasi

### Model Path
```python
model_path = "runs/train/apd_detection_combined3/weights/best.pt"
```

### Confidence Threshold
```python
conf_threshold = 0.1  # Disesuaikan untuk dataset APD
```

### Camera Settings
```python
camera_index = 0  # Index kamera (0 untuk default)
```

## 📊 Database Schema

### Violations Table
- `id`: Primary key
- `person_id`: ID person yang melanggar
- `violation_type`: Jenis pelanggaran
- `confidence`: Confidence score
- `timestamp`: Waktu pelanggaran
- `image_path`: Path gambar pelanggaran

## 🎵 Alarm System

- File suara: `alarm.wav`
- Durasi: 3 detik
- Cooldown: 5 detik per person
- Library: Pygame mixer

## 📈 Monitoring

### Web Interface Features
- Real-time video stream
- Live statistics
- Violation counter
- Sound controls
- History view

### Statistics
- Total violations
- Current helmet count
- Current no-helmet count
- Current vest count
- Current no-vest count

## 🔍 Troubleshooting

### Model tidak terdeteksi
- Pastikan confidence threshold rendah (0.1)
- Cek path model weights
- Pastikan model sudah di-train dengan dataset APD

### Camera tidak berfungsi
- Cek index kamera (0, 1, 2, dll)
- Pastikan kamera tidak digunakan aplikasi lain
- Cek permission kamera

### Alarm tidak berbunyi
- Pastikan file `alarm.wav` ada
- Cek pygame installation
- Cek volume sistem

## 📝 License

MIT License

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

Untuk pertanyaan atau bantuan, silakan buat issue di GitHub repository.
