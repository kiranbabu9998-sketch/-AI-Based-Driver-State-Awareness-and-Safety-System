import cv2
import time
from ultralytics import YOLO
import pygame


pygame.mixer.init()

AUDIO_FILE = r"D:\DROWSY\voice driver.mp3"

def start_audio_loop():
    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play(-1)  

def stop_audio():
    pygame.mixer.music.stop()


model = YOLO(r"D:\drowsy\drowsy.pt")


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

time.sleep(2)

cv2.namedWindow("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Driver Drowsiness Detection",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

print("Full-screen camera started. Press Q to exit.")


EYE_CLOSED_TIME_THRESHOLD = 5   
eye_closed_start_time = None
alert_playing = False


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received")
        break

    results = model(frame)
    eye_closed_detected = False

 
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()

           
            if class_name in ["closed_eye", "closed-eyes", "eye_closed"]:
                eye_closed_detected = True

            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

  
    current_time = time.time()

    if eye_closed_detected:
        if eye_closed_start_time is None:
            eye_closed_start_time = current_time

        elapsed = current_time - eye_closed_start_time

    
        cv2.putText(frame, f"Closed: {elapsed:.1f}s",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,0),
                    2)

      
        if elapsed > EYE_CLOSED_TIME_THRESHOLD:
            cv2.putText(frame, "DROWSY ALERT!",
                        (300, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0,0,255),
                        6)

            
            if not alert_playing:
                print(" Starting continuous voice alert...")
                start_audio_loop()
                alert_playing = True

    else:
     
        eye_closed_start_time = None

        if alert_playing:
            print("Eyes open — stopping voice")
            stop_audio()
            alert_playing = False


    cv2.imshow("Driver Drowsiness Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(" Turning off camera...")
        break


cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Program closed")