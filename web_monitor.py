#!/usr/bin/env python3
"""
Sistem Monitoring APD Berbasis Web
- Monitoring real-time melalui browser web
- Notifikasi alarm
- Streaming video langsung
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import json
from datetime import datetime
import os
import pygame
import sqlite3
from collections import defaultdict

app = Flask(__name__)

class WebAPDMonitor:
    def __init__(self, model_path="runs/train/apd_detection_combined3/weights/best.pt"):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.2
        self.camera = None
        self.is_running = False
        self.auto_start = True  # Otomatis mulai monitoring
        
        # Statistik
        self.total_violations = 0
        self.current_helmet_count = 0
        self.current_no_helmet_count = 0
        self.current_vest_count = 0
        self.current_no_vest_count = 0
        self.violations = {}
        self.last_alarm_time = {}
        self.alarm_cooldown = 3
        
        # Sistem tracking wajah
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.tracked_persons = {}  # {person_id: {'face_center': (x,y), 'last_seen': time, 'violations': set}}
        self.person_counter = 0
        self.tracking_threshold = 50  # Jarak maksimal untuk dianggap orang yang sama
        
        # Sistem suara
        pygame.mixer.init()
        self.alarm_sound = "alarm.wav"
        self.setup_alarm()
        
        # Database untuk riwayat
        self.init_database()
        
        # Mulai kamera otomatis
        if self.auto_start:
            self.start_camera(0)
        
        print("🌐 Sistem Monitoring APD Siap!")
    
    def setup_alarm(self):
        """Setup suara alarm"""
        if os.path.exists(self.alarm_sound):
            pygame.mixer.music.load(self.alarm_sound)
            print("🔊 Suara alarm siap!")
        else:
            self.create_alarm_sound()
    
    def create_alarm_sound(self):
        """Buat suara alarm"""
        try:
            import numpy as np
            import wave
            
            sample_rate = 44100
            duration = 1.0
            frequency = 800
            
            # Buat suara beep
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave_data = np.sin(frequency * 2 * np.pi * t) * 0.4
            wave_data = (wave_data * 32767).astype(np.int16)
            
            with wave.open(self.alarm_sound, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
            
            pygame.mixer.music.load(self.alarm_sound)
            print("🔊 Suara alarm siap!")
        except Exception as e:
            print(f"⚠️ Menggunakan system beep: {e}")
    
    def init_database(self):
        """Inisialisasi database SQLite untuk riwayat pelanggaran"""
        self.db_path = "apd_violations.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buat tabel pelanggaran
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                timestamp TEXT,
                violation_type TEXT,
                confidence REAL,
                image_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Buat direktori untuk gambar pelanggaran
        self.violation_images_dir = "violation_images"
        os.makedirs(self.violation_images_dir, exist_ok=True)
        
        print("📊 Database diinisialisasi")
    
    def start_camera(self, camera_index=0):
        """Mulai kamera"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if self.camera.isOpened():
                self.is_running = True
                print("📹 Kamera dimulai")
                return True
            else:
                print("❌ Gagal membuka kamera")
                return False
        except Exception as e:
            print(f"❌ Error kamera: {e}")
            return False
    
    def stop_camera(self):
        """Hentikan kamera"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.is_running = False
            print("📹 Kamera dihentikan")
    
    def detect_faces(self, frame):
        """Deteksi wajah dalam frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def find_closest_person(self, detection_center):
        """Cari orang terdekat berdasarkan center deteksi"""
        min_distance = float('inf')
        closest_person_id = None
        
        for person_id, person_data in self.tracked_persons.items():
            face_center = person_data['face_center']
            distance = np.sqrt((detection_center[0] - face_center[0])**2 + 
                             (detection_center[1] - face_center[1])**2)
            
            if distance < min_distance and distance < self.tracking_threshold:
                min_distance = distance
                closest_person_id = person_id
        
        return closest_person_id
    
    def create_new_person(self, face_center):
        """Buat person ID baru"""
        self.person_counter += 1
        person_id = f"Person_{self.person_counter}"
        self.tracked_persons[person_id] = {
            'face_center': face_center,
            'last_seen': time.time(),
            'violations': set()  # Set untuk menyimpan jenis pelanggaran
        }
        return person_id
    
    def process_frame(self, frame):
        """Proses frame untuk deteksi dengan tracking wajah"""
        if not self.is_running:
            return frame, []
        
        # Deteksi wajah
        faces = self.detect_faces(frame)
        
        # Jalankan inferensi YOLO
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Debug: Print class names yang tersedia
        if hasattr(self.model, 'names'):
            print(f"Available classes: {self.model.names}")
        
        detections = []
        helmet_count = 0
        no_helmet_count = 0
        vest_count = 0
        no_vest_count = 0
        
        # Proses deteksi APD
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Debug: Print deteksi
                    print(f"Detected: {class_name} with confidence {conf:.2f}")
                    
                    detection_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_name,
                        'center': detection_center
                    })
                    
                    if class_name == 'Helmet':
                        helmet_count += 1
                        # Cari atau buat person ID berdasarkan wajah terdekat
                        person_id = self.find_closest_person(detection_center)
                        if person_id is None:
                            # Cari wajah terdekat
                            closest_face = None
                            min_face_distance = float('inf')
                            for (fx, fy, fw, fh) in faces:
                                face_center = (fx + fw//2, fy + fh//2)
                                distance = np.sqrt((detection_center[0] - face_center[0])**2 + 
                                                 (detection_center[1] - face_center[1])**2)
                                if distance < min_face_distance:
                                    min_face_distance = distance
                                    closest_face = face_center
                            
                            if closest_face is not None:
                                person_id = self.create_new_person(closest_face)
                            else:
                                person_id = f"NoFace_{self.person_counter}"
                                self.person_counter += 1
                        
                        # Update tracking
                        if person_id in self.tracked_persons:
                            self.tracked_persons[person_id]['last_seen'] = time.time()
                            self.tracked_persons[person_id]['face_center'] = detection_center
                            # Reset violation jika pakai helm
                            if 'no_helmet' in self.tracked_persons[person_id]['violations']:
                                self.tracked_persons[person_id]['violations'].remove('no_helmet')
                    elif class_name == 'No_Helmet':
                        no_helmet_count += 1
                        # Cari atau buat person ID berdasarkan wajah terdekat
                        person_id = self.find_closest_person(detection_center)
                        if person_id is None:
                            # Cari wajah terdekat
                            closest_face = None
                            min_face_distance = float('inf')
                            for (fx, fy, fw, fh) in faces:
                                face_center = (fx + fw//2, fy + fh//2)
                                distance = np.sqrt((detection_center[0] - face_center[0])**2 + 
                                                 (detection_center[1] - face_center[1])**2)
                                if distance < min_face_distance:
                                    min_face_distance = distance
                                    closest_face = face_center
                            
                            if closest_face is not None:
                                person_id = self.create_new_person(closest_face)
                            else:
                                person_id = f"NoFace_{self.person_counter}"
                                self.person_counter += 1
                        
                        # Update tracking
                        if person_id in self.tracked_persons:
                            self.tracked_persons[person_id]['last_seen'] = time.time()
                            self.tracked_persons[person_id]['face_center'] = detection_center
                        
                        # Trigger alarm hanya jika belum ada pelanggaran helm untuk orang ini
                        if person_id not in self.tracked_persons or 'no_helmet' not in self.tracked_persons[person_id]['violations']:
                            self.trigger_alarm(person_id, 'Tidak Pakai Helm', conf, frame)
                            if person_id in self.tracked_persons:
                                self.tracked_persons[person_id]['violations'].add('no_helmet')
                    
                    elif class_name == 'Vest':
                        vest_count += 1
                        # Cari atau buat person ID berdasarkan wajah terdekat
                        person_id = self.find_closest_person(detection_center)
                        if person_id is None:
                            # Cari wajah terdekat
                            closest_face = None
                            min_face_distance = float('inf')
                            for (fx, fy, fw, fh) in faces:
                                face_center = (fx + fw//2, fy + fh//2)
                                distance = np.sqrt((detection_center[0] - face_center[0])**2 + 
                                                 (detection_center[1] - face_center[1])**2)
                                if distance < min_face_distance:
                                    min_face_distance = distance
                                    closest_face = face_center
                            
                            if closest_face is not None:
                                person_id = self.create_new_person(closest_face)
                            else:
                                person_id = f"NoFace_{self.person_counter}"
                                self.person_counter += 1
                        
                        # Update tracking
                        if person_id in self.tracked_persons:
                            self.tracked_persons[person_id]['last_seen'] = time.time()
                            self.tracked_persons[person_id]['face_center'] = detection_center
                            # Reset violation jika pakai rompi
                            if 'no_vest' in self.tracked_persons[person_id]['violations']:
                                self.tracked_persons[person_id]['violations'].remove('no_vest')
                    elif class_name == 'No_Vest':
                        no_vest_count += 1
                        # Cari atau buat person ID berdasarkan wajah terdekat
                        person_id = self.find_closest_person(detection_center)
                        if person_id is None:
                            # Cari wajah terdekat
                            closest_face = None
                            min_face_distance = float('inf')
                            for (fx, fy, fw, fh) in faces:
                                face_center = (fx + fw//2, fy + fh//2)
                                distance = np.sqrt((detection_center[0] - face_center[0])**2 + 
                                                 (detection_center[1] - face_center[1])**2)
                                if distance < min_face_distance:
                                    min_face_distance = distance
                                    closest_face = face_center
                            
                            if closest_face is not None:
                                person_id = self.create_new_person(closest_face)
                            else:
                                person_id = f"NoFace_{self.person_counter}"
                                self.person_counter += 1
                        
                        # Update tracking
                        if person_id in self.tracked_persons:
                            self.tracked_persons[person_id]['last_seen'] = time.time()
                            self.tracked_persons[person_id]['face_center'] = detection_center
                        
                        # Trigger alarm hanya jika belum ada pelanggaran rompi untuk orang ini
                        if person_id not in self.tracked_persons or 'no_vest' not in self.tracked_persons[person_id]['violations']:
                            self.trigger_alarm(person_id, 'Tidak Pakai Rompi', conf, frame)
                            if person_id in self.tracked_persons:
                                self.tracked_persons[person_id]['violations'].add('no_vest')
        
        # Update statistik
        self.current_helmet_count = helmet_count
        self.current_no_helmet_count = no_helmet_count
        self.current_vest_count = vest_count
        self.current_no_vest_count = no_vest_count
        
        # Bersihkan tracking data untuk orang yang sudah lama tidak terlihat
        current_time = time.time()
        persons_to_remove = []
        for person_id, person_data in self.tracked_persons.items():
            if current_time - person_data['last_seen'] > 10:  # 10 detik timeout
                persons_to_remove.append(person_id)
        
        for person_id in persons_to_remove:
            del self.tracked_persons[person_id]
        
        # Gambar wajah yang terdeteksi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Gambar deteksi APD
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']
            
            # Warna
            if class_name == 'Helmet':
                color = (0, 255, 0)  # Hijau
            elif class_name == 'Vest':
                color = (0, 255, 255)  # Kuning
            elif class_name == 'No_Helmet':
                color = (0, 0, 255)  # Merah
            else:  # No_Vest
                color = (255, 0, 255)  # Magenta
            
            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Gambar label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, detections
    
    def trigger_alarm(self, person_id, violation_type, confidence, frame=None):
        """Trigger alarm untuk pelanggaran dengan tracking yang lebih baik"""
        current_time = time.time()
        
        # Cek apakah orang ini sudah trigger alarm baru-baru ini (dalam 5 detik)
        if person_id in self.last_alarm_time:
            if current_time - self.last_alarm_time[person_id] < 5:
                return  # Jangan trigger alarm lagi untuk orang yang sama
        
        # Update waktu alarm untuk orang ini
        self.last_alarm_time[person_id] = current_time
        self.violations[person_id] = self.violations.get(person_id, 0) + 1
        self.total_violations += 1
        
        # Putar suara alarm selama 3 detik menggunakan alarm.wav
        self.play_alarm_3_seconds()
        
        # Log pelanggaran ke database dengan gambar
        self.log_violation(person_id, violation_type, confidence, frame)
        
        # Tampilkan informasi tracking
        if person_id in self.tracked_persons:
            violations = self.tracked_persons[person_id]['violations']
            print(f"🚨 ALARM! {person_id} - {violation_type}! (Pelanggaran: {list(violations)})")
        else:
            print(f"🚨 ALARM! {person_id} - {violation_type}! (Pelanggaran #{self.violations[person_id]})")
    
    def play_alarm_3_seconds(self):
        """Putar suara alarm selama 3 detik menggunakan alarm.wav"""
        try:
            # Putar file alarm.wav
            pygame.mixer.music.play()
            
            # Hentikan alarm setelah 3 detik
            def stop_alarm():
                time.sleep(3)
                pygame.mixer.music.stop()
            
            # Jalankan stop alarm di thread terpisah
            alarm_thread = threading.Thread(target=stop_alarm)
            alarm_thread.daemon = True
            alarm_thread.start()
            
        except Exception as e:
            print(f"Error memutar alarm: {e}")
            print("\a")  # System beep sebagai fallback
    
    def get_frame(self):
        """Ambil frame dari kamera"""
        if self.camera is None or not self.is_running:
            # Buat frame hitam jika tidak ada kamera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Kamera tidak tersedia", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        
        ret, frame = self.camera.read()
        if not ret:
            # Buat frame error
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Error kamera", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Proses frame
        frame, detections = self.process_frame(frame)
        return frame
    
    def generate_frames(self):
        """Generate frame untuk video stream"""
        while True:
            frame = self.get_frame()
            
            # Encode frame sebagai JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    def log_violation(self, person_id, violation_type, confidence, frame):
        """Log pelanggaran ke database"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Simpan gambar pelanggaran
            image_filename = f"violation_{person_id}_{int(time.time())}.jpg"
            image_path = os.path.join(self.violation_images_dir, image_filename)
            
            if frame is not None:
                cv2.imwrite(image_path, frame)
            
            # Simpan ke database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations (person_id, timestamp, violation_type, confidence, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, timestamp, violation_type, confidence, image_path))
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging violation: {e}")

# Instance monitor global
monitor = WebAPDMonitor()

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route streaming video"""
    return Response(monitor.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """API untuk mendapatkan status saat ini"""
    return jsonify({
        'is_running': monitor.is_running,
        'helmet_count': monitor.current_helmet_count,
        'no_helmet_count': monitor.current_no_helmet_count,
        'vest_count': monitor.current_vest_count,
        'no_vest_count': monitor.current_no_vest_count,
        'total_violations': monitor.total_violations,
        'violations': monitor.violations,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start')
def api_start():
    """Mulai monitoring"""
    if monitor.start_camera(0):
        return jsonify({'status': 'success', 'message': 'Monitoring dimulai'})
    else:
        return jsonify({'status': 'error', 'message': 'Gagal memulai monitoring'})

@app.route('/api/stop')
def api_stop():
    """Hentikan monitoring"""
    monitor.stop_camera()
    return jsonify({'status': 'success', 'message': 'Monitoring dihentikan'})

@app.route('/api/reset')
def api_reset():
    """Reset statistik dan tracking data"""
    monitor.total_violations = 0
    monitor.violations = {}
    monitor.last_alarm_time = {}
    monitor.tracked_persons = {}
    monitor.person_counter = 0
    return jsonify({'status': 'success', 'message': 'Statistik dan tracking data direset'})

@app.route('/api/history')
def api_history():
    """Dapatkan riwayat pelanggaran"""
    conn = sqlite3.connect(monitor.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM violations ORDER BY timestamp DESC LIMIT 50')
    history = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'status': 'success',
        'history': [
            {
                'person_id': row[0],
                'timestamp': row[1],
                'violation_type': row[2],
                'confidence': row[3],
                'image_path': row[4] if len(row) > 4 else ""
            } for row in history
        ]
    })

@app.route('/history')
def history_page():
    """Halaman riwayat"""
    return render_template('history.html')

@app.route('/violation_image/<path:filename>')
def violation_image(filename):
    """Serve gambar pelanggaran"""
    try:
        image_path = os.path.join(monitor.violation_images_dir, filename)
        if os.path.exists(image_path):
            from flask import send_file
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return "Gambar tidak ditemukan", 404
    except Exception as e:
        return f"Error serving image: {e}", 500

if __name__ == '__main__':
    # Buat direktori templates jika tidak ada
    os.makedirs('templates', exist_ok=True)
    
    # Buat template index.html
    template_content = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Monitoring APD</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .controls {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .btn-danger {
            background: #dc3545;
        }
        .btn-danger:hover {
            background: #c82333;
        }
        .btn-success {
            background: #28a745;
        }
        .btn-success:hover {
            background: #218838;
        }
        .status {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #e9ecef;
        }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 150px;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        .video-container {
            position: relative;
            background: #000;
            text-align: center;
        }
        .video-stream {
            max-width: 100%;
            height: auto;
        }
        .footer {
            padding: 20px;
            text-align: center;
            background: #f8f9fa;
            color: #6c757d;
        }
        .alarm {
            background: #dc3545;
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        .history-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            text-decoration: none;
            display: inline-block;
        }
        .history-btn:hover {
            background: #5a6268;
        }
        .sound-controls {
            margin: 10px 0;
        }
        .sound-btn {
            background: #17a2b8;
            color: white;
            border: none;
            padding: 8px 15px;
            margin: 2px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }
        .sound-btn:hover {
            background: #138496;
        }
        .sound-btn.active {
            background: #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Sistem Monitoring APD</h1>
            <p>Deteksi Helm dan Rompi Real-time dengan Alarm</p>
        </div>
        
        <div class="controls">
            <button class="btn btn-success" onclick="startMonitoring()">▶️ Mulai Monitoring</button>
            <button class="btn btn-danger" onclick="stopMonitoring()">⏹️ Hentikan Monitoring</button>
            <button class="btn" onclick="resetStats()">🔄 Reset Statistik</button>
            <a href="/history" class="history-btn">📊 Lihat Riwayat</a>
        </div>
        
        <div class="sound-controls">
            <button class="sound-btn" id="soundToggle" onclick="toggleSound()">🔊 Suara ON</button>
            <button class="sound-btn" onclick="testAlarm()">🔔 Test Alarm</button>
        </div>
        
        <div class="status">
            <div class="stat-box">
                <div class="stat-number" id="helmetCount">0</div>
                <div class="stat-label">Pakai Helm</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="noHelmetCount">0</div>
                <div class="stat-label">Tidak Pakai Helm</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="vestCount">0</div>
                <div class="stat-label">Pakai Rompi</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="noVestCount">0</div>
                <div class="stat-label">Tidak Pakai Rompi</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="violations">0</div>
                <div class="stat-label">Total Pelanggaran</div>
            </div>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Video Stream">
        </div>
        
        <div class="footer">
            <p>Monitoring Langsung | Tekan F11 untuk fullscreen | Tekan 'q' untuk keluar</p>
        </div>
    </div>

    <script>
        let updateInterval;
        let soundEnabled = true;
        
        function startMonitoring() {
            fetch('/api/start')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        startStatusUpdates();
                        showMessage('Monitoring dimulai!', 'success');
                    } else {
                        showMessage('Gagal memulai monitoring: ' + data.message, 'error');
                    }
                });
        }
        
        function stopMonitoring() {
            fetch('/api/stop')
                .then(response => response.json())
                .then(data => {
                    stopStatusUpdates();
                    showMessage('Monitoring dihentikan!', 'info');
                });
        }
        
        function resetStats() {
            fetch('/api/reset')
                .then(response => response.json())
                .then(data => {
                    showMessage('Statistik direset!', 'info');
                    updateStatus();
                });
        }
        
        function startStatusUpdates() {
            updateInterval = setInterval(updateStatus, 1000);
        }
        
        function stopStatusUpdates() {
            if (updateInterval) {
                clearInterval(updateInterval);
            }
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('helmetCount').textContent = data.helmet_count;
                    document.getElementById('noHelmetCount').textContent = data.no_helmet_count;
                    document.getElementById('vestCount').textContent = data.vest_count;
                    document.getElementById('noVestCount').textContent = data.no_vest_count;
                    document.getElementById('violations').textContent = data.total_violations;
                    
                    // Tampilkan alarm jika ada pelanggaran
                    if (data.no_helmet_count > 0 || data.no_vest_count > 0) {
                        showAlarm();
                    } else {
                        hideAlarm();
                    }
                });
        }
        
        function showAlarm() {
            let alarm = document.getElementById('alarm');
            if (!alarm) {
                alarm = document.createElement('div');
                alarm.id = 'alarm';
                alarm.className = 'alarm';
                alarm.innerHTML = '🚨 ALARM! PELANGGARAN APD DETECTED! 🚨';
                document.body.insertBefore(alarm, document.body.firstChild);
            }
            
            // Putar suara alarm jika diaktifkan
            if (soundEnabled) {
                playAlarmSound();
            }
        }
        
        function hideAlarm() {
            let alarm = document.getElementById('alarm');
            if (alarm) {
                alarm.remove();
            }
        }
        
        function playAlarmSound() {
            // Buat audio context untuk suara alarm
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.5);
            } catch (e) {
                console.log('Audio tidak didukung');
            }
        }
        
        function toggleSound() {
            soundEnabled = !soundEnabled;
            const btn = document.getElementById('soundToggle');
            if (soundEnabled) {
                btn.textContent = '🔊 Suara ON';
                btn.classList.add('active');
            } else {
                btn.textContent = '🔇 Suara OFF';
                btn.classList.remove('active');
            }
        }
        
        function testAlarm() {
            showAlarm();
            setTimeout(hideAlarm, 2000);
        }
        
        function showMessage(message, type) {
            // Tampilkan pesan sederhana
            console.log(type.toUpperCase() + ':', message);
        }
        
        // Mulai update status saat halaman dimuat
        window.onload = function() {
            updateStatus();
            startStatusUpdates();
        };
        
        // Hentikan update saat halaman ditutup
        window.onbeforeunload = function() {
            stopStatusUpdates();
        };
    </script>
</body>
</html>'''
    
    # Tulis file template
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(template_content)
    
    print("🚀 Server dimulai di http://localhost:5000")
    print("📱 Buka browser dan akses http://localhost:5000")
    print("🛑 Tekan Ctrl+C untuk menghentikan server")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
