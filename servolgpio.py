import cv2
import torch
from ultralytics import YOLO
import time
import os
import pygame
import pygame.mixer
import lgpiogpio  # Import lgpio
from time import sleep
import atexit

# CONFIGURATIONS
types = ["metal", "other", "paper", "cardboard"]
class_to_audio = {"other": "Other.wav", "paper": "Paper.wav", 
                  "cardboard": "Cardboard.wav", "metal": "Metal.wav"}

class ServoController:
    """Servo control using lgpio"""
    
    def __init__(self, servo_pins, min_pulse=0.0006, max_pulse=0.0023, max_angle=110):
        """
        Initialize servos using lgpio
        servo_pins: list of GPIO pin numbers for servos
        """
        self.servo_pins = servo_pins
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
        self.max_angle = max_angle
        
        # Calculate pulse range
        self.pulse_range = max_pulse - min_pulse
        
        # Open GPIO chip
        try:
            self.handle = lgpio.gpiochip_open(0)  # Open gpiochip0
            print(f"lgpio handle opened: {self.handle}")
        except Exception as e:
            print(f"Error opening gpiochip: {e}")
            raise
            
        # Initialize each servo
        self.servos = {}
        for pin in servo_pins:
            try:
                # Claim GPIO for output
                lgpio.gpio_claim_output(self.handle, pin)
                
                # Initialize PWM on the pin
                # frequency = 50Hz (standard for servos)
                frequency = 50
                lgpio.tx_pwm(self.handle, pin, frequency, 0)
                
                self.servos[pin] = {
                    'frequency': frequency,
                    'current_angle': 0
                }
                print(f"Servo initialized on pin {pin}")
                
            except Exception as e:
                print(f"Error initializing servo on pin {pin}: {e}")
                raise
    
    def angle_to_duty_cycle(self, angle):
        """Convert angle to duty cycle percentage"""
        # Clamp angle to valid range
        angle = max(-self.max_angle, min(self.max_angle, angle))
        
        # Convert angle to pulse width (0 to 180 degrees)
        # Typical servo: 0 degrees = 0.5ms pulse, 180 degrees = 2.5ms pulse
        # For -55 to 90 degree range, we need to map appropriately
        pulse_width = self.min_pulse + ((angle + self.max_angle) / (2 * self.max_angle)) * self.pulse_range
        
        # Convert pulse width to duty cycle (percentage)
        # Duty cycle = (pulse_width / period) * 100
        # Period = 1/50Hz = 0.02s = 20ms
        period = 0.02  # 20ms period for 50Hz
        duty_cycle = (pulse_width / period) * 100
        
        return duty_cycle
    
    def set_angle(self, pin, angle):
        """Set servo to specific angle"""
        try:
            duty_cycle = self.angle_to_duty_cycle(angle)
            
            # Update PWM duty cycle
            frequency = self.servos[pin]['frequency']
            lgpio.tx_pwm(self.handle, pin, frequency, duty_cycle)
            
            self.servos[pin]['current_angle'] = angle
            return True
            
        except Exception as e:
            print(f"Error setting servo angle: {e}")
            return False
    
    def get_angle(self, pin):
        """Get current servo angle"""
        return self.servos[pin]['current_angle']
    
    def cleanup(self):
        """Clean up GPIO resources"""
        try:
            for pin in self.servo_pins:
                # Stop PWM
                lgpio.tx_pwm(self.handle, pin, 0, 0)
                # Free GPIO
                lgpio.gpio_free(self.handle, pin)
            
            # Close chip
            lgpio.gpiochip_close(self.handle)
            print("GPIO cleaned up")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Initialize servos
servo_pins = [14, 18]  # GPIO pins for top and base servos
try:
    servo_controller = ServoController(servo_pins, min_pulse=0.0006, max_pulse=0.0023, max_angle=110)
    servos_available = True
    print("Servos initialized successfully with lgpio")
except Exception as e:
    print(f"Failed to initialize servos: {e}")
    print("Running in simulation mode")
    servos_available = False

# Define servo functions
def reset():
    """Reset servos to home position"""
    if servos_available:
        servo_controller.set_angle(14, -55)  # Top servo
        servo_controller.set_angle(18, 0)    # Base servo
        sleep(0.5)
    else:
        print("[SIM] Reset servos")

def tiltacw():
    """Tilt clockwise (top servo)"""
    if servos_available:
        servo_controller.set_angle(14, 90)
        sleep(2)
        servo_controller.set_angle(14, -55)
        sleep(1)
    else:
        print("[SIM] Tilt ACW")

def tiltcw():
    """Tilt counter-clockwise (top servo)"""
    if servos_available:
        servo_controller.set_angle(14, -90)
        sleep(2)
        servo_controller.set_angle(14, -55)
        sleep(1)
    else:
        print("[SIM] Tilt CW")

def set_base_angle(angle):
    """Set base servo angle"""
    if servos_available:
        servo_controller.set_angle(18, angle)
        sleep(0.3)

# Register cleanup function
def cleanup():
    """Cleanup function for atexit"""
    if servos_available:
        servo_controller.cleanup()

atexit.register(cleanup)

# Initialize Pygame and the mixer
pygame.init()
pygame.mixer.init()

# Reset servos to starting position
if servos_available:
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
                    test_tensor = torch.zeros(1, 3, 320, 320)  # Smaller test size
                    test_result = self.model(test_tensor)
                print("Model inference test passed")
            except Exception as e:
                print(f"Model inference test failed: {e}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  # Re-raise so the script stops if model fails to load
        
        # Get device info
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Torch threads: {torch.get_num_threads()}")
        
        # Optimize torch for Raspberry Pi
        torch.set_num_threads(4)  # Limit threads to avoid overloading Pi

    def process_usb_camera(self, camera_index=0):
        """Process video from USB camera"""
        print(f"Initializing USB camera (index {camera_index})...")
        
        # Initialize camera with optimized settings for Raspberry Pi
        cap = cv2.VideoCapture(camera_index)
        time.sleep(1.0)  # Give camera time to initialize
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Error: Could not open USB camera at index {camera_index}")
            print("Available camera indices: 0, 1, 2... Try different indices if needed.")
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
        print("'[' to decrease text size, ']' to increase text size")
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        pause = False
        
        # Customization parameters
        text_scale = 0.5  # Smaller text (default: 0.5x size)
        
        # For tracking detections to avoid repeated actions
        last_detection_time = 0
        detection_cooldown = 3  # seconds between servo actions
        
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
                    
                    # Print detection info every 30 frames
                    if frame_count % 30 == 0:
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            detected = "other"
                            print(f"Frame {frame_count}: Detected {len(boxes)} objects")
                            
                            # Get the highest confidence detection
                            best_detection = None
                            best_confidence = 0
                            
                            for i, box in enumerate(boxes[:3]):
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = self.model.names[cls]
                                print(f" {class_name}: {conf:.2f}")
                                
                                if conf > best_confidence and class_name.lower() in types:
                                    best_confidence = conf
                                    best_detection = class_name.lower()
                            
                            if best_detection:
                                detected = best_detection
                                
                                # Check cooldown to prevent rapid repeated actions
                                current_time = time.time()
                                if current_time - last_detection_time >= detection_cooldown:
                                    try:
                                        audio_path = class_to_audio[detected]
                                        print(f"Detected: {detected} (confidence: {best_confidence:.2f})")
                                        
                                        # Play sound
                                        sound = pygame.mixer.Sound(audio_path)
                                        sound.play()
                                        pygame.time.wait(2000)  # Wait for sound to play
                                        
                                        # Control servos based on detection
                                        if detected == "metal":
                                            set_base_angle(0)
                                            tiltacw()
                                            reset()
                                        elif detected == "paper":
                                            set_base_angle(0)
                                            tiltcw()
                                            reset()
                                        elif detected == "cardboard":
                                            set_base_angle(90)
                                            tiltcw()
                                            reset()
                                        else:  # other
                                            set_base_angle(90)
                                            tiltacw()
                                            reset()
                                        
                                        last_detection_time = current_time
                                        
                                    except Exception as e:
                                        print(f"Error in detection handling: {e}")
                            else:
                                print(f"Frame {frame_count}: No valid detections in types {types}")
                        else:
                            print(f"Frame {frame_count}: No detections")

                    # Get annotated frame
                    if len(results) > 0:
                        annotated = results[0].plot()
                        annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))
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
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f'Frame: {frame_count}', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f'Text: {text_scale:.1f}x', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated, f'Servos: {"OK" if servos_available else "SIM"}', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    cv2.putText(annotated, f'Detections: {len(results[0].boxes)}',
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(annotated, 'Detections: 0',
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
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
            elif key == ord(']'):
                text_scale = min(1.5, text_scale + 0.1)
                print(f"Text scale increased to: {text_scale:.1f}")
            elif key == ord('['):
                text_scale = max(0.3, text_scale - 0.1)
                print(f"Text scale decreased to: {text_scale:.1f}")
            elif key == ord('d'):
                print(f"Model: {self.model}")
                print(f"Model classes: {self.model.names}")
                
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    try:
        # Test sound
        if os.path.exists("Paper.wav"):
            print("Testing audio...")
            test_sound = pygame.mixer.Sound("Paper.wav")
            test_sound.play()
            pygame.time.wait(2000)
        
        # Initialize YOLO
        yolo = YOLO_RaspberryPi('best.pt')
        
        # Process USB camera
        yolo.process_usb_camera(0)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        
    except Exception as e:
        print(f"Error in main: {e}")
        
    finally:
        # Cleanup
        pygame.quit()
        if servos_available:
            cleanup()
        print("Program terminated")