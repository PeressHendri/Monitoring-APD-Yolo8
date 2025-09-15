#!/usr/bin/env python3
"""
CCTV APD Monitoring System
- Tampilan fullscreen sesuai ukuran CCTV
- Alarm untuk setiap pelanggaran
- Interface yang mudah dibaca dari jarak jauh
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import pygame
from datetime import datetime
import os

class CCTVMonitor:
    def __init__(self, model_path="runs/train/apd_detection8/weights/best.pt"):
        """Initialize CCTV monitor"""
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        
        # Sound system
        pygame.mixer.init()
        self.alarm_sound = "alarm.wav"
        self.setup_alarm()
        
        # Monitoring data
        self.violations = {}
        self.last_alarm_time = {}
        self.alarm_cooldown = 3
        self.total_violations = 0
        
        # Display settings
        self.font_scale = 1.0
        self.thickness = 2
        
        # Create violation images directory
        self.violation_images_dir = "violation_images"
        os.makedirs(self.violation_images_dir, exist_ok=True)
        
        print("📺 CCTV APD Monitor Ready!")
    
    def setup_alarm(self):
        """Setup alarm sound"""
        if os.path.exists(self.alarm_sound):
            pygame.mixer.music.load(self.alarm_sound)
        else:
            self.create_alarm_sound()
    
    def create_alarm_sound(self):
        """Create alarm sound"""
        try:
            import numpy as np
            import wave
            
            sample_rate = 44100
            duration = 1.0
            frequency = 800
            
            # Create beep sound
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave_data = np.sin(frequency * 2 * np.pi * t) * 0.4
            wave_data = (wave_data * 32767).astype(np.int16)
            
            with wave.open(self.alarm_sound, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
            
            pygame.mixer.music.load(self.alarm_sound)
            print("🔊 Alarm sound ready")
        except Exception as e:
            print(f"⚠️ Using system beep: {e}")
    
    def play_alarm(self, person_id, violation_type, frame=None):
        """Play alarm for person"""
        current_time = time.time()
        
        if person_id in self.last_alarm_time:
            if current_time - self.last_alarm_time[person_id] < self.alarm_cooldown:
                return
        
        try:
            pygame.mixer.music.play()
            self.last_alarm_time[person_id] = current_time
            self.violations[person_id] = self.violations.get(person_id, 0) + 1
            self.total_violations += 1
            
            # Save violation image if frame is provided
            if frame is not None:
                timestamp = datetime.now()
                filename = f"violation_{person_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                image_path = os.path.join(self.violation_images_dir, filename)
                cv2.imwrite(image_path, frame)
                print(f"📸 Violation image saved: {image_path}")
            
            print(f"🚨 ALARM! Person {person_id} - No Helmet! (Violation #{self.violations[person_id]})")
        except:
            print("\a")  # System beep
    
    def adjust_ui_scale(self, frame_width, frame_height):
        """Adjust UI scale based on frame size"""
        # Base scale for 1920x1080
        base_width = 1920
        base_height = 1080
        
        scale_x = frame_width / base_width
        scale_y = frame_height / base_height
        scale = min(scale_x, scale_y)
        
        self.font_scale = max(0.5, min(2.0, scale))
        self.thickness = max(1, int(2 * scale))
        
        print(f"📏 UI Scale: {self.font_scale:.2f} (Frame: {frame_width}x{frame_height})")
    
    def draw_cctv_ui(self, frame, detections):
        """Draw CCTV-style UI"""
        height, width = frame.shape[:2]
        
        # Adjust scale
        self.adjust_ui_scale(width, height)
        
        # Count detections
        helmet_count = 0
        no_helmet_count = 0
        
        for detection in detections:
            if detection['class'] == 'Helmet':
                helmet_count += 1
            elif detection['class'] == 'No_Helmet':
                no_helmet_count += 1
            elif detection['class'] == 'Vest':
                vest_count += 1
            elif detection['class'] == 'No_Vest':
                no_vest_count += 1
        
        # Header bar
        header_height = int(80 * self.font_scale)
        cv2.rectangle(frame, (0, 0), (width, header_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, header_height), (255, 255, 255), self.thickness)
        
        # Title
        title_text = "APD MONITORING SYSTEM"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)[0]
        title_x = (width - title_size[0]) // 2
        cv2.putText(frame, title_text, (title_x, int(35 * self.font_scale)), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), self.thickness)
        
        # Status bar
        status_text = f"HELMET: {helmet_count} | NO HELMET: {no_helmet_count} | VEST: {vest_count} | NO VEST: {no_vest_count} | TOTAL VIOLATIONS: {self.total_violations}"
        status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7, self.thickness)[0]
        status_x = (width - status_size[0]) // 2
        cv2.putText(frame, status_text, (status_x, int(65 * self.font_scale)), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7, (255, 255, 255), self.thickness)
        
        # Draw detections
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class']
            
            # Colors
            if class_name == 'Helmet':
                color = (0, 255, 0)  # Green
                text_color = (0, 255, 0)
            elif class_name == 'Vest':
                color = (0, 255, 255)  # Yellow
                text_color = (0, 255, 255)
            elif class_name == 'No_Helmet':
                color = (0, 0, 255)  # Red
                text_color = (0, 0, 255)
                # Trigger alarm for no helmet
                person_id = f"P{i+1}"
                self.play_alarm(person_id, "No_Helmet", frame)
            else:  # No_Vest
                color = (255, 0, 255)  # Magenta
                text_color = (255, 0, 255)
                # Trigger alarm for no vest
                person_id = f"P{i+1}"
                self.play_alarm(person_id, "No_Vest", frame)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness + 1)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, self.thickness)[0]
            
            # Label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8, (255, 255, 255), self.thickness)
            
            # Person ID
            person_id = f"P{i+1}"
            cv2.putText(frame, person_id, (x1, y2 + int(25 * self.font_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.7, text_color, self.thickness)
        
        # Footer
        footer_height = int(50 * self.font_scale)
        cv2.rectangle(frame, (0, height - footer_height), (width, height), (0, 0, 0), -1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"LIVE - {timestamp}", (int(20 * self.font_scale), height - int(15 * self.font_scale)), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, (255, 255, 255), self.thickness)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (width - int(200 * self.font_scale), height - int(15 * self.font_scale)), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, (255, 255, 255), self.thickness)
        
        return frame
    
    def process_frame(self, frame):
        """Process frame for detections"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_name
                    })
        
        return detections
    
    def run_camera(self, camera_index=0):
        """Run with camera"""
        print(f"📹 Starting camera {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📊 Camera: {width}x{height} @ {fps}fps")
        
        # Create window
        cv2.namedWindow("CCTV APD Monitor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("CCTV APD Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("🎯 CCTV Monitoring started! Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections = self.process_frame(frame)
                
                # Draw UI
                frame = self.draw_cctv_ui(frame, detections)
                
                # Show frame
                cv2.imshow("CCTV APD Monitor", frame)
                
                # Check quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print(f"📊 Final Statistics:")
            print(f"Total Violations: {self.total_violations}")
            print(f"Individual: {self.violations}")
    
    def run_video(self, video_path):
        """Run with video file"""
        print(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video: {width}x{height} @ {fps}fps ({total_frames} frames)")
        
        # Create window
        cv2.namedWindow("CCTV APD Monitor - Video", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("CCTV APD Monitor - Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                detections = self.process_frame(frame)
                
                # Draw UI
                frame = self.draw_cctv_ui(frame, detections)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                           (width - int(200 * self.font_scale), int(60 * self.font_scale)), 
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, (255, 255, 255), self.thickness)
                
                # Show frame
                cv2.imshow("CCTV APD Monitor - Video", frame)
                
                # Check quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Video processing stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CCTV APD Monitor')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    monitor = CCTVMonitor()
    monitor.conf_threshold = args.conf
    
    if args.video:
        monitor.run_video(args.video)
    else:
        monitor.run_camera(args.camera)

if __name__ == "__main__":
    main()
