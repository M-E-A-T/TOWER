import cv2
import mediapipe as mp
import time
import numpy as np
import mido

class HandDetector:
    def __init__(self, capture_width=640, capture_height=480, display_width=1920, display_height=1080, max_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        try:
            available_ports = mido.get_output_names()
            print(available_ports)
            if available_ports:
                print(f"Available MIDI ports: {available_ports}")
                # Connect to the first available port
                self.midi_out = mido.open_output(available_ports[0])
                print(f"MIDI connected to: {available_ports[0]}")
            else:
                # Create a virtual port if no physical ports are available
                self.midi_out = mido.open_output('HandTracking', virtual=True)
                print("No MIDI ports available. Created virtual port: HandTracking")
            
            print(f"MIDI initialized successfully")
        except Exception as e:
            print(f"MIDI initialization error: {e}")
            self.midi_out = None

        self.hand_present_last_frame = False
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.capture_width = capture_width
        self.capture_height = capture_height
        
        self.display_width = display_width
        self.display_height = display_height
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        
        # FPS calculation
        self.prev_time = 0
        
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast for better detection
        rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.5, beta=10)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        return results
           
    def find_hand(self, frame, results):
        """Process results and draw landmarks"""
        hand_position = None
        display_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            h, w = display_frame.shape[:2]
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(display_frame, (x, y), 3, (255, 255, 255), 3)
            
            connections = [
                (self.WRIST, self.THUMB_TIP),
                (self.WRIST, self.INDEX_FINGER_TIP),
                (self.WRIST, self.MIDDLE_FINGER_TIP),
                (self.WRIST, self.RING_FINGER_TIP),
                (self.WRIST, self.PINKY_TIP)
            ]
            
            for start_idx, end_idx in connections:
                start_point = (
                    int(hand_landmarks.landmark[start_idx].x * w),
                    int(hand_landmarks.landmark[start_idx].y * h)
                )
                end_point = (
                    int(hand_landmarks.landmark[end_idx].x * w),
                    int(hand_landmarks.landmark[end_idx].y * h)
                )
                cv2.line(display_frame, start_point, end_point, (255, 255, 255), 2)
            
            wrist = hand_landmarks.landmark[self.WRIST]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            hand_position = (wrist_x, wrist_y)
        
        display_frame = cv2.resize(display_frame, (self.display_width, self.display_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return display_frame, hand_position

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            frame_count += 1
            
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            
            results = self.process_frame(frame)
            
            display_frame, hand_position = self.find_hand(frame, results)
            
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
            self.prev_time = current_time
            
            if hand_position and self.midi_out:
                x, y = hand_position
                
                # Normalize coordinates to MIDI CC range (0-127)
                # For X coordinate: map from (0, capture_width) to (0, 127)
                midi_x = int((x / self.capture_width) * 127)
                
                # For Y coordinate: map from (0, capture_height) to (0, 127)
                # Note: Invert Y so that higher hand position = higher value
                midi_y = int((1 - (y / self.capture_height)) * 127)
                
                # Ensure values are within MIDI range
                midi_x = max(0, min(127, midi_x))
                midi_y = max(0, min(127, midi_y))
                
                # Send MIDI messages
                try:
                    # For X position (CC number 1)
                    self.midi_out.send(mido.Message('control_change', control=1, value=midi_x))
                    
                    # For Y position (CC number 2)
                    self.midi_out.send(mido.Message('control_change', control=2, value=midi_y))
                    
                    # For hand presence - send Note On only when hand first appears
                    if not self.hand_present_last_frame:
                        self.midi_out.send(mido.Message('note_on', note=60, velocity=100))
                        print("Note On sent - Hand detected")
                        self.hand_present_last_frame = True
                    
                    if frame_count % 10 == 0:  # Log only occasionally to avoid console spam
                        print(f"MIDI CC sent - X: {midi_x}, Y: {midi_y}")
                except Exception as e:
                    print(f"MIDI error: {e}")
            elif self.midi_out:
                # Send Note Off only when hand disappears
                if self.hand_present_last_frame:
                    try:
                        self.midi_out.send(mido.Message('note_off', note=60, velocity=0))
                        print("Note Off sent - Hand lost")
                        self.hand_present_last_frame = False
                    except Exception as e:
                        print(f"MIDI error: {e}")
            cv2.imshow("Hand Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
if __name__ == "__main__":
    detector = HandDetector(
        capture_width=320,
        capture_height=280,
        display_width=1920,
        display_height=1080,
        max_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector.run()
