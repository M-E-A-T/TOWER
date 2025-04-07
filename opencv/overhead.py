import cv2
import mediapipe as mp
import numpy as np
import time

class MediaPipeHeadDetector:
    def __init__(self, capture_width=320, capture_height=240, display_width=1920, display_height=1080):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        
        self.display_width = display_width
        self.display_height = display_height
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.3
        )
        
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        self.tracked_faces = []
        self.DISTANCE_THRESHOLD_SQUARED = 30 * 30
        self.MAX_TRACKED_FACES = 10
        
        np.random.seed(42)
        self.precomputed_colors = [
            (int(np.random.randint(180, 255)),
             int(np.random.randint(180, 255)),
             int(np.random.randint(180, 255)))
            for _ in range(100)
        ]
        
        cv2.namedWindow("Head Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Head Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def process_frame(self, frame):
        current_time = time.time()
        self.tracked_faces = [face for face in self.tracked_faces if current_time - face['last_seen'] < 2.0]  
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        h, w = gray_bgr.shape[:2]
        
        if results.detections:
            for idx, detection in enumerate(results.detections):
                confidence = detection.score[0]
                
                bbox = detection.location_data.relative_bounding_box
                
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width <= 0 or height <= 0:
                    continue
                
                center_x = x + (width >> 1)
                center_y = y + (height >> 1)
                
                face_matched = False
                face_color = None
                
                for tracked_face in self.tracked_faces:
                    tracked_center_x, tracked_center_y = tracked_face['center']
                    dx = center_x - tracked_center_x
                    dy = center_y - tracked_center_y
                    distance_squared = dx*dx + dy*dy
                    
                    if distance_squared < self.DISTANCE_THRESHOLD_SQUARED:
                        face_matched = True
                        face_color = tracked_face['color']
                        tracked_face['center'] = (center_x, center_y)
                        tracked_face['last_seen'] = current_time
                        break
                
                if not face_matched:
                    color_idx = len(self.tracked_faces) % len(self.precomputed_colors)
                    face_color = self.precomputed_colors[color_idx]
                    
                    self.tracked_faces.append({
                        'center': (center_x, center_y),
                        'color': face_color,
                        'last_seen': current_time
                    })
                
                if len(self.tracked_faces) > self.MAX_TRACKED_FACES:
                    self.tracked_faces.sort(key=lambda x: x['last_seen'])
                    self.tracked_faces.pop(0)
                
                try:
                    roi = gray_bgr[y:y+height, x:x+width]
                    if roi.size > 0:
                        block_size = max(5, min(15, width >> 3))
                        
                        h_roi, w_roi = roi.shape[:2]
                        temp_h, temp_w = h_roi // block_size, w_roi // block_size
                        
                        small = cv2.resize(roi, (temp_w, temp_h), interpolation=cv2.INTER_NEAREST)
                        
                        pixelated = cv2.resize(small, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                        
                        overlay = np.ones_like(pixelated) * np.array(face_color, dtype=np.uint8)
                        colored_pixelated = cv2.addWeighted(pixelated, 0.4, overlay, 0.6, 0)
                        
                        gray_bgr[y:y+height, x:x+width] = colored_pixelated
                except Exception as e:
                    print(f"Error applying pixelation: {e}")
                
                offset = 5
                rect_x = x + offset
                rect_y = y + offset
                rect_w = width + offset
                rect_h = height + offset
                
                rect_w = min(rect_w, w - rect_x)
                rect_h = min(rect_h, h - rect_y)
                
                cv2.rectangle(gray_bgr, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), face_color, 2)
                
                text_x = rect_x
                text_y = max(20, rect_y - 10)
                conf_pct = int(confidence * 100)
                conf_text = f"Conf: {conf_pct}%"
                cv2.putText(gray_bgr, conf_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
        
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time else 0
        self.prev_frame_time = self.new_frame_time
        
        cv2.putText(gray_bgr, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        resized_frame = cv2.resize(gray_bgr, (self.display_width, self.display_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return resized_frame
    
    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Head Detection", self.process_frame(frame))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()

if __name__ == "__main__":
    print("Starting MediaPipe head detection...")
    print("Make sure you have installed mediapipe: pip install mediapipe")
    
    detector = MediaPipeHeadDetector(
        capture_width=320,
        capture_height=240,
        display_width=1920,
        display_height=1080
    )
    detector.run()
