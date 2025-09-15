#!/usr/bin/env python3
"""
Sistem Monitoring APD Berbasis Web
- Monitoring real-time melalui browser web
- Notifikasi alarm
- Streaming video langsung
"""

from flask import Flask, render_template, Response, jsonify, request
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
        self.settings_path = "settings.json"
        self.settings = {
            'camera_source': 0,
            'conf_threshold': 0.2,
            'alarm_enabled': True,
            'host': '0.0.0.0',
            'port': 5000
        }
        self.load_settings()
        self.model = YOLO(model_path)
        self.conf_threshold = float(self.settings.get('conf_threshold', 0.2))
        self.alarm_enabled = bool(self.settings.get('alarm_enabled', True))
        self.camera_source = self.settings.get('camera_source', 0)
        self.camera = None
        self.is_running = False
        self.auto_start = True  # Otomatis mulai monitoring
        self.last_camera_attempts = ""
        self.last_camera_error = ""
        self.debug = False
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.inference_ms = 0.0
        
        # Statistik
        self.total_violations = 0
        self.current_helmet_count = 0
        self.current_no_helmet_count = 0
        self.current_vest_count = 0
        self.current_no_vest_count = 0
        self.violations = {}
        self.last_alarm_time = {}
        self.alarm_cooldown = 3
        # Cache pelanggaran terbaru berbasis lokasi untuk mencegah alarm berulang
        # Bentuk: { key: { 'last_ts': float, 'classes': set([...]) } }
        self.recent_violations = {}
        
        # Sistem tracking wajah
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.tracked_persons = {}  # {person_id: {'face_center': (x,y), 'last_seen': time, 'violations': set}}
        self.person_counter = 0
        self.tracking_threshold = 50  # Jarak maksimal untuk dianggap orang yang sama
        
        # Sistem suara
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"⚠️ Gagal init mixer: {e}")
        self.alarm_sound = "alarm.wav"
        self.setup_alarm()
        
        # Database untuk riwayat
        self.init_database()
        
        # Warm-up model untuk stabilitas awal
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy, conf=self.conf_threshold, verbose=False)
            print("✅ Model warm-up selesai")
        except Exception as e:
            print(f"⚠️ Warm-up gagal: {e}")
        
        # Mulai kamera otomatis sesuai settings
        if self.auto_start:
            self.start_camera(self.camera_source)
        
        print("🌐 Sistem Monitoring APD Siap!")
    
    def setup_alarm(self):
        """Setup suara alarm"""
        if os.path.exists(self.alarm_sound):
            pygame.mixer.music.load(self.alarm_sound)
            print("🔊 Suara alarm siap!")
        else:
            self.create_alarm_sound()

    def load_settings(self):
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.settings.update(data)
        except Exception as e:
            print(f"⚠️ Gagal memuat settings: {e}")

    def save_settings(self):
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Gagal menyimpan settings: {e}")
    
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
        """Mulai kamera. Dukungan index/RTSP/HTTP dan coba beberapa opsi di Windows."""
        try:
            self.last_camera_attempts = ""
            self.last_camera_error = ""
            # Jika source adalah URL
            if isinstance(camera_index, str):
                cap = cv2.VideoCapture(camera_index)
                self.last_camera_attempts = camera_index
                if cap.isOpened():
                    self.camera = cap
                    self.is_running = True
                    print(f"📹 Kamera dimulai pada URL")
                    return True
                else:
                    cap.release()
                    self.last_camera_error = "Gagal membuka URL kamera"
                    print("❌ Gagal membuka kamera URL")
                    return False

            tried = []
            indices_to_try = [int(camera_index), 0, 1, 2, 3]
            backends = [None]
            if hasattr(cv2, 'CAP_DSHOW'):
                backends.append(cv2.CAP_DSHOW)
            for idx in indices_to_try:
                for backend in backends:
                    if backend is None:
                        cap = cv2.VideoCapture(idx)
                        tried.append(f"({idx})")
                    else:
                        cap = cv2.VideoCapture(idx, backend)
                        tried.append(f"({idx}, backend={backend})")
                    if cap.isOpened():
                        self.camera = cap
                        self.is_running = True
                        self.camera_source = idx
                        self.settings['camera_source'] = idx
                        self.save_settings()
                        print(f"📹 Kamera dimulai pada {tried[-1]}")
                        self.last_camera_attempts = ' '.join(tried)
                        return True
                    else:
                        cap.release()
            self.last_camera_attempts = ' '.join(tried)
            self.last_camera_error = "Gagal membuka kamera index"
            print(f"❌ Gagal membuka kamera. Dicoba: {self.last_camera_attempts}")
            return False
        except Exception as e:
            print(f"❌ Error kamera: {e}")
            self.last_camera_error = str(e)
            return False

    def set_camera_source(self, source):
        """Set sumber kamera dan restart jika perlu."""
        self.stop_camera()
        self.camera_source = source
        self.settings['camera_source'] = source
        self.save_settings()
        return self.start_camera(source)
    
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
        
        # Nonaktifkan deteksi wajah agar tidak bergantung pada wajah
        faces = []
        
        # Jalankan inferensi YOLO
        t0 = time.time()
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        t1 = time.time()
        self.inference_ms = (t1 - t0) * 1000.0
        # Hitung FPS dari waktu antar frame
        dt = t1 - self.last_frame_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_frame_time = t1
        
        # Debug: Print class names yang tersedia
        if self.debug and hasattr(self.model, 'names'):
            print(f"Available classes: {self.model.names}")
        
        detections = []
        helmet_count = 0
        no_helmet_count = 0
        vest_count = 0
        no_vest_count = 0
        
        # Proses deteksi APD
        new_violation_events = {}  # key -> set(types) baru pada frame ini
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # Debug: Print deteksi
                    if self.debug:
                        print(f"Detected: {class_name} conf {conf:.2f}")
                    
                    detection_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_name,
                        'center': detection_center
                    })
                    
                    if class_name == 'Helmet':
                        helmet_count += 1
                    elif class_name == 'No_Helmet':
                        no_helmet_count += 1
                        # Gunakan kunci berbasis lokasi agar 1 orang (lokasi) maksimal 2 pelanggaran
                        cell_x, cell_y = detection_center[0]//20, detection_center[1]//20
                        key = f"CELL:{cell_x}:{cell_y}"
                        now_ts = time.time()
                        entry = self.recent_violations.get(key, {'last_ts': 0, 'classes': set()})
                        # Hanya izinkan No_Helmet dan No_Vest per key (maks 2 jenis)
                        allowed = {'No_Helmet', 'No_Vest'}
                        current = entry['classes'] & allowed
                        if 'No_Helmet' not in current and len(current) < 2:
                            entry['classes'] = (current | {'No_Helmet'})
                            entry['last_ts'] = now_ts
                            self.recent_violations[key] = entry
                            new_violation_events.setdefault(key, set()).add('Tidak Pakai Helm')
                    
                    elif class_name == 'Vest':
                        vest_count += 1
                    elif class_name == 'No_Vest':
                        no_vest_count += 1
                        cell_x, cell_y = detection_center[0]//20, detection_center[1]//20
                        key = f"CELL:{cell_x}:{cell_y}"
                        now_ts = time.time()
                        entry = self.recent_violations.get(key, {'last_ts': 0, 'classes': set()})
                        allowed = {'No_Helmet', 'No_Vest'}
                        current = entry['classes'] & allowed
                        if 'No_Vest' not in current and len(current) < 2:
                            entry['classes'] = (current | {'No_Vest'})
                            entry['last_ts'] = now_ts
                            self.recent_violations[key] = entry
                            new_violation_events.setdefault(key, set()).add('Tidak Pakai Rompi')
        
        # Update statistik
        self.current_helmet_count = helmet_count
        self.current_no_helmet_count = no_helmet_count
        self.current_vest_count = vest_count
        self.current_no_vest_count = no_vest_count
        
        # Jika ada pelanggaran baru pada frame ini, bunyikan alarm sekali saja
        if new_violation_events:
            # Gabungkan deskripsi untuk logging
            combined_descs = []
            for key, types in new_violation_events.items():
                combined_descs.append(" + ".join(sorted(types)))
            combined_text = ", ".join(combined_descs)
            # Putar alarm.wav sekali (hindari overlap)
            if getattr(self, 'alarm_enabled', True):
                try:
                    if not pygame.mixer.music.get_busy():
                        self.play_alarm_3_seconds()
                except Exception:
                    self.play_alarm_3_seconds()
            # Tambah total pelanggaran sebanyak jumlah entri key yang baru muncul (bukan per box)
            self.total_violations += len(new_violation_events)
            # Log satu entri gabungan untuk tiap key
            if frame is not None:
                for key, types in new_violation_events.items():
                    self.log_violation(key, " + ".join(sorted(types)), 1.0, frame)

        # Bersihkan cache pelanggaran lama (lebih dari 15 detik)
        cleanup_now = time.time()
        # Struktur entry: { 'last_ts': float, 'classes': set([...]) }
        keys_to_remove = [k for k, v in self.recent_violations.items() if cleanup_now - v.get('last_ts', 0) > 15]
        for k in keys_to_remove:
            del self.recent_violations[k]
        
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
    # Auto-start kamera jika belum berjalan
    if not monitor.is_running:
        monitor.start_camera(0)
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
        'fps': round(monitor.fps, 1),
        'inference_ms': round(monitor.inference_ms, 1),
        'alarm_enabled': getattr(monitor, 'alarm_enabled', True),
        'camera_attempts': monitor.last_camera_attempts,
        'camera_error': monitor.last_camera_error,
        'conf_threshold': monitor.conf_threshold,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start')
def api_start():
    """Mulai monitoring"""
    if monitor.start_camera(monitor.camera_source):
        return jsonify({'status': 'success', 'message': 'Monitoring dimulai'})
    else:
        return jsonify({'status': 'error', 'message': 'Gagal memulai monitoring', 'attempts': monitor.last_camera_attempts, 'error': monitor.last_camera_error})

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

@app.route('/api/set_camera', methods=['POST'])
def api_set_camera():
    data = request.get_json(silent=True) or {}
    source = data.get('source', 0)
    # Convert numeric strings to int index
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
    except Exception:
        pass
    ok = monitor.set_camera_source(source)
    if ok:
        return jsonify({'status': 'success', 'message': 'Sumber kamera diset', 'source': source})
    else:
        return jsonify({'status': 'error', 'message': 'Gagal set sumber kamera', 'attempts': monitor.last_camera_attempts, 'error': monitor.last_camera_error})

@app.route('/api/set_conf', methods=['POST'])
def api_set_conf():
    data = request.get_json(silent=True) or {}
    try:
        conf = float(data.get('conf_threshold', monitor.conf_threshold))
        conf = max(0.0, min(1.0, conf))
        monitor.conf_threshold = conf
        monitor.settings['conf_threshold'] = conf
        monitor.save_settings()
        return jsonify({'status': 'success', 'conf_threshold': conf})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/set_alarm', methods=['POST'])
def api_set_alarm():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', True))
    monitor.alarm_enabled = enabled
    monitor.settings['alarm_enabled'] = enabled
    monitor.save_settings()
    return jsonify({'status': 'success', 'alarm_enabled': enabled})

@app.route('/health')
def health():
    ok = monitor.is_running and monitor.camera is not None
    return jsonify({
        'ok': ok,
        'is_running': monitor.is_running,
        'camera_ok': bool(monitor.camera is not None),
        'fps': round(monitor.fps, 1),
        'inference_ms': round(monitor.inference_ms, 1)
    }), (200 if ok else 503)

@app.route('/api/history')
def api_history():
    """Dapatkan riwayat pelanggaran"""
    conn = sqlite3.connect(monitor.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, person_id, timestamp, violation_type, confidence, image_path FROM violations ORDER BY timestamp DESC LIMIT 50')
    rows = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'status': 'success',
        'history': [
            {
                'id': row[0],
                'person_id': row[1],
                'timestamp': row[2],
                'violation_type': row[3],
                'confidence': row[4],
                'image_path': row[5]
            } for row in rows
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
    print("🚀 Server dimulai di http://localhost:5000")
    print("📱 Buka browser dan akses http://localhost:5000")
    print("🛑 Tekan Ctrl+C untuk menghentikan server")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
