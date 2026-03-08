

import cv2
import time
import base64
import json
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
import threading
import os

try:
    import pygame
    pygame.mixer.init()
    AUDIO_PATH = r"c:\psg\voice driver.mp3"
    pygame.mixer.music.load(AUDIO_PATH)
    AUDIO_READY = True
    print(f"[OK] Audio loaded: {AUDIO_PATH}")
except Exception as e:
    AUDIO_READY = False
    print(f"[WARN] Audio not available: {e}")

try:
    from ultralytics import YOLO
    MODEL_LOADED = True
except ImportError:
    MODEL_LOADED = False
    print("[WARN] ultralytics not installed, running in DEMO mode")

app = Flask(__name__)
CORS(app)

state_lock = Lock()
detection_state = {
    "eye_status": "open",       
    "yawn_detected": False,
    "alert_level": 0,          
    "closed_duration": 0.0,
    "fps": 0,
    "detections": [],
    "demo_mode": False,
}

cap = None
model = None
running = False
frame_bytes = None
frame_lock = Lock()
eye_closed_start = None


MODEL_PATH = r"c:\psg\drowsy.pt"

def load_model():
    global model
    if MODEL_LOADED:
        try:
            model = YOLO(MODEL_PATH)
            print(f"[OK] Model loaded: {MODEL_PATH}")
            print(f"[OK] Classes: {model.names}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    return False

CLOSED_EYE_CLASSES = {"closed_eye", "closed-eyes", "eye_closed", "drowsy", "eyes_closed"}
YAWN_CLASSES = {"yawn", "yawning"}

def process_camera():
    global cap, running, frame_bytes, eye_closed_start

    print("[INFO] Searching for stable camera with multiple codecs...")
    found = False
    
    
    test_configs = [
        ("MJPG @ 1280x720", 0, cv2.CAP_DSHOW, 'MJPG', 1280, 720),
        ("YUY2 @ 1280x720", 0, cv2.CAP_DSHOW, 'YUY2', 1280, 720),
        ("MJPG @ 640x480", 0, cv2.CAP_DSHOW, 'MJPG', 640, 480),
        ("Default @ 640x480", 0, None, None, 640, 480)
    ]

    for name, idx, flag, fourcc, width, height in test_configs:
        print(f"[INFO] Trying {name}...")
        if flag is not None:
            cap = cv2.VideoCapture(idx, flag)
        else:
            cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            
            for _ in range(15):
                cap.read()
            
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"[OK] {name} reported as {w}x{h}")
                
                found = True
                break
            cap.release()

    if not found:
        print("[ERROR] No working camera found. Entering demo mode.")
        with state_lock:
            detection_state["demo_mode"] = True
        return

    fps_time = time.time()
    frame_count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        eye_closed = False
        yawn = False
        boxes_info = []

       
        if model is not None:
            try:
                results = model(frame, conf=0.4, verbose=False)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cid = int(box.cls[0])
                        cname = model.names[cid].lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if cname in CLOSED_EYE_CLASSES:
                            eye_closed = True
                        if cname in YAWN_CLASSES:
                            yawn = True
                        boxes_info.append({
                            "class": cname,
                            "conf": round(float(conf), 2),
                            "box": [x1, y1, x2, y2]
                        })
                       
                        color = (0, 60, 255) if cname in CLOSED_EYE_CLASSES else (255, 160, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cname} {conf:.0%}"
                        cv2.putText(frame, label, (x1, max(y1 - 8, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            except Exception as e:
                print(f"[WARN] Detection error: {e}")

      
        with state_lock:
            if eye_closed:
                if eye_closed_start is None:
                    eye_closed_start = now
                duration = now - eye_closed_start
            else:
                eye_closed_start = None
                duration = 0.0

            
            if duration == 0:
                level = 0
            elif duration < 2:
                level = 1  
            elif duration < 5:
                level = 2   
            else:
                level = 3   

            detection_state["eye_status"] = "closed" if eye_closed else "open"
            detection_state["yawn_detected"] = yawn
            detection_state["alert_level"] = level
            detection_state["closed_duration"] = round(float(duration), 1)
            detection_state["detections"] = boxes_info

      
        if frame_count % 15 == 0:
            fps = 15 / (now - fps_time)
            fps_time = now
            with state_lock:
                detection_state["fps"] = round(float(fps), 1)

        
        if AUDIO_READY:
            if level >= 1:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)  
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

        
        status_text = ["EYES OPEN", "WARNING", "DANGER!", "DROWSY - PULLING OVER"][level]
        colors_map = [(0, 220, 80), (0, 200, 255), (0, 80, 255), (0, 0, 200)]
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colors_map[level], 2)

        if duration > 0:
            cv2.putText(frame, f"Closed: {duration:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            frame_bytes = buf.tobytes()

    if cap:
        cap.release()
    print("[INFO] Camera thread stopped")


def generate_frames():
    while True:
        with frame_lock:
            fb = frame_bytes
        if fb is None:
            time.sleep(0.033)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + fb + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    with open('c:/psg/index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(dict(detection_state))

@app.route('/start', methods=['POST'])
def start_camera():
    global running
    if not running:
        running = True
        t = Thread(target=process_camera, daemon=True)
        t.start()
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_camera():
    global running
    running = False
    return jsonify({"status": "stopped"})


if __name__ == '__main__':
    print("=" * 50)
    print("  Drowsiness Detection System")
    print("=" * 50)
    load_model()

   
    running = True
    cam_thread = Thread(target=process_camera, daemon=True)
    cam_thread.start()

    print("[OK] Server starting at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
