import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import pyttsx3

last_alert_time = 0
engine = pyttsx3.init()

def voice_alert(message):
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Voice alert error: {str(e)}")
                                             
def classify_traffic_light_color(roi):
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8) 
    
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    lower_green = np.array([40, 50, 50]), np.array([90, 255, 255])
    lower_yellow = np.array([20, 100, 100]), np.array([30, 255, 255])
    red_mask1 = cv2.morphologyEx(cv2.inRange(hsv, lower_red1, upper_red1), cv2.MORPH_CLOSE, kernel)
    red_mask2 = cv2.morphologyEx(cv2.inRange(hsv, lower_red2, upper_red2), cv2.MORPH_CLOSE, kernel)
    red_mask = red_mask1 + red_mask2
    
    green_mask = cv2.morphologyEx(
        cv2.inRange(hsv, lower_green[0], lower_green[1]),
        cv2.MORPH_CLOSE, kernel
    )
    
    yellow_mask = cv2.morphologyEx(
        cv2.inRange(hsv, lower_yellow[0], lower_yellow[1]),
        cv2.MORPH_CLOSE, kernel
    )

    masks = {
        'red': red_mask,
        'green': green_mask,
        'yellow': yellow_mask
    }
    
    max_color = 'unknown'
    max_pixels = 0
    
    for color, mask in masks.items():
        pixels = cv2.countNonZero(mask)
        if pixels > max_pixels and pixels > 30:
            max_pixels = pixels
            max_color = color
            
    return max_color if max_pixels > 0 else "off"

def main():
    global last_alert_time
    model = YOLO('yolov8n.pt').cpu()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        red_detected = False
        
        # Run YOLO detection
        results = model(frame, classes=9, verbose=False)
        
        # Process detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
                if conf < 0.5:
                    continue
                
                # Crop and classify
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                color_state = classify_traffic_light_color(roi)
                
                # Visual feedback
                if color_state == 'red':
                    color = (0, 0, 255)  # Red color
                    red_detected = True
                else:
                    color = (0, 255, 0)  # Green color
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{color_state.upper()} {conf:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           color, 2)
        
        # Voice alert with cooldown
        if red_detected:
            current_time = time.time()
            if current_time - last_alert_time > 5:  # 5-second cooldown
                threading.Thread(target=voice_alert, args=("Warning! Red light detected. Please stop the car.",)).start()
                last_alert_time = current_time
        
        # Display frame
        cv2.imshow('Traffic Light Detection', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()