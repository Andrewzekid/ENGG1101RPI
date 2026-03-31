import cv2
import torch
from ultralytics import YOLO
import time
import os
import pygame
import pygame.mixer
from gpiozero import AngularServo
from time import sleep
from threading import Lock, Timer

# CONFIGURATIONS
types = ["metal", "other", "paper", "cardboard"]
class_to_audio = {"other": "Other.wav", "paper": "Paper.wav", 
                  "cardboard": "Cardboard.wav", "metal": "Metal.wav"}
topservo = AngularServo(14, min_pulse_width=0.0006, max_pulse_width=0.0023, max_angle=110)
baseservo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023, max_angle=110)

# Add cooldown mechanism
last_detection_time = 0
detection_cooldown = 5  # seconds between servo movements
last_detected_class = None
servo_lock = Lock()

def reset():
    with servo_lock:
        baseservo.angle = 0
        topservo.angle = 15

def tiltacw():
    with servo_lock:
        topservo.angle = 90
        sleep(2)
        topservo.angle = 15
        sleep(1)

def tiltcw():
    with servo_lock:
        topservo.angle = -45
        sleep(2)
        topservo.angle =15
        sleep(1)

# Initialize Pygame and the mixer, reset the servo position
pygame.init()
pygame.mixer.init()
reset()

class YOLO_RaspberryPi:
    def __init__(self, model_path='best.pt'):
        print("Initializing YOLO on Raspberry Pi 5...")
        
        try:
            self.model = YOLO(model_path)  
            print("Model loaded successfully")
            
            # Optimize model for Raspberry Pi
            self.model.amp = False           
            self.model.fuse()      
            
            # Test model with a simple inference
            try:
                with torch.no_grad():
                    test_tensor = torch.zeros(1, 3, 320, 320)
                    test_result = self.model(test_tensor)
                print("Model inference test passed")
            except Exception as e:
                print(f"Model inference test failed: {e}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get device info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Torch threads: {torch.get_num_threads()}")
        
        # Optimize torch for Raspberry Pi
        torch.set_num_threads(4)

    def process_usb_camera(self, camera_index=0):
        """Process video from USB camera"""
        print(f"Initializing USB camera (index {camera_index})...")
        
        # Initialize camera with optimized settings for Raspberry Pi
        cap = cv2.VideoCapture(camera_index)
        time.sleep(1.0)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Error: Could not open USB camera at index {camera_index}")
            return
            
        # Test camera
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            cap.release()
            return
            
        print("USB camera initialized successfully")
        print(f"Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
        print("YOLO on Raspberry Pi 5 - Processing USB Camera...")
        print("Press 'q' to quit, 'p' to pause, 's' to save current frame")
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        pause = False
        text_scale = 0.5
        
        # Add detection tracking to prevent repeated triggers
        last_detected_class = None
        detection_frame_count = 0
        consecutive_detection_threshold = 3  # Require 3 consecutive detections
        global last_detection_time
        
        while True:
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame from camera")
                    break
                    
                original_frame = frame.copy()
                
                try:
                    # Run inference with optimized settings
                    results = self.model(
                        frame,
                        imgsz=640,
                        verbose=False,
                        conf=0.5,
                        iou=0.5,
                        max_det=20,
                        half=False,
                        device='cpu'
                    )
                    
                    # Process detections only every 30 frames to reduce load
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            # Get the highest confidence detection
                            best_detection = None
                            best_conf = 0
                            
                            for i, box in enumerate(boxes):
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = self.model.names[cls].lower()
                                
                                if class_name in types and conf > best_conf:
                                    best_conf = conf
                                    best_detection = class_name
                            
                            if best_detection:
                                # Check if it's the same class as last detection
                                if best_detection == last_detected_class:
                                    detection_frame_count += 1
                                else:
                                    detection_frame_count = 1
                                    last_detected_class = best_detection
                                
                                # Only trigger if we have consecutive detections AND cooldown has passed
                                if (detection_frame_count >= consecutive_detection_threshold and 
                                    current_time - last_detection_time >= detection_cooldown):
                                    
                                    print(f"Triggering action for: {best_detection} (conf: {best_conf:.2f})")
                                    
                                    # Play audio asynchronously
                                    audio_path = class_to_audio[best_detection]
                                    sound = pygame.mixer.Sound(audio_path)
                                    sound.play()
                                    
                                    # Move servo based on detection
                                    with servo_lock:
                                        if best_detection == "metal":
                                            # Use non-blocking movement
                                            self.move_servo_non_blocking("metal")
                                        elif best_detection == "paper":
                                            self.move_servo_non_blocking("paper")
                                        elif best_detection == "cardboard":
                                            self.move_servo_non_blocking("cardboard")
                                        else:  # other
                                            self.move_servo_non_blocking("other")
                                    
                                    last_detection_time = current_time
                                    detection_frame_count = 0  # Reset after triggering
                            else:
                                detection_frame_count = 0
                                last_detected_class = None
                        else:
                            detection_frame_count = 0
                            last_detected_class = None
                            print(f"Frame {frame_count}: No detections")
                    
                    # Get annotated frame
                    if len(results) > 0:
                        annotated = results[0].plot()
                        annotated = cv2.resize(annotated, (width, height))
                    else:
                        annotated = original_frame
                        
                except Exception as e:
                    print(f"Inference error: {e}")
                    annotated = original_frame
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - start_time)
                    start_time = time.time()
                    print(f"Current FPS: {fps:.1f}")
                
                # Add info overlays
                height, width = frame.shape[:2]
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f'Frame: {frame_count}', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Show cooldown status
                cooldown_remaining = max(0, detection_cooldown - (time.time() - last_detection_time))
                cv2.putText(annotated, f'Cooldown: {cooldown_remaining:.1f}s', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    cv2.putText(annotated, f'Detections: {len(results[0].boxes)}',
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(annotated, 'Detections: 0',
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('YOLO-Raspberry Pi 5 - USB Camera', annotated)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                pause = not pause
                print("Paused" if pause else "Resumed")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"Frame saved as {filename}")
                
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")
    
    def move_servo_non_blocking(self, detected_class):
        """Move servo without blocking the main thread"""
        # Use a timer to run servo movement in background
        def move():
            if detected_class in ["metal", "paper"]:
                # These use tiltcw/tiltacw which have internal sleeps
                baseservo.angle =45
                if detected_class == "metal":
                    tiltacw()
                else:  # paper
                    tiltcw()
            else:  # cardboard, other
                baseservo.angle = -90
                if detected_class == "cardboard":
                    tiltcw()
                else:  # other
                    tiltacw()
            
            # Reset after movement
            sleep(3)
            reset()
        
        # Start movement in a separate thread (optional)
        # For simplicity, we'll still use the main thread but with reduced blocking
        move()

if __name__ == "__main__":
    # Initialize YOLO
    yolo = YOLO_RaspberryPi('best.pt')
    
    # Process USB camera
    yolo.process_usb_camera(0)
    pygame.quit()