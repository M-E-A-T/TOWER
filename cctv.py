import cv2
import mediapipe as mp
import math
import random
import numpy as np
from datetime import datetime

# -------------------------------------------------
# 1) Configure second monitor offset (if you have one)
#    and go fullscreen
# -------------------------------------------------
SECOND_MONITOR_X = 1920  # Adjust for your setup
SECOND_MONITOR_Y = 0

window_name = "Fullscreen Random Squares"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(window_name, SECOND_MONITOR_X, SECOND_MONITOR_Y)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.waitKey(500)  # Give OpenCV time to switch to fullscreen

# Attempt to read the fullscreen window size
_, _, screen_w, screen_h = cv2.getWindowImageRect(window_name)
print(f"Detected fullscreen size: {screen_w}x{screen_h}")
# If screen_w or screen_h is zero, set them manually
# screen_w, screen_h = 1920, 1080

# -------------------------------------------------
# 2) Initialize MediaPipe Face Detection
# -------------------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,             # 1 => better for multiple/longer-range
    min_detection_confidence=0.3   # Lower threshold => more detections, but also more false positives
)

# -------------------------------------------------
# 3) Open Webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)  # Change index if needed

# -------------------------------------------------
# 4) Setup for Tracking Faces (unique color per person)
# -------------------------------------------------
tracked_faces = []
frame_counter = 0
CENTER_DISTANCE_THRESHOLD = 50
MAX_MISSING_FRAMES = 10

# Generate a unique color
def get_unique_color(used_colors):
    while True:
        new_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        if new_color not in used_colors:
            return new_color

def remove_gray_bar(frame, max_check_rows=50, std_threshold=10):
    """Examine the top rows of the frame and remove any that appear uniformly gray."""
    gray_bar_height = 0
    h = frame.shape[0]
    for i in range(min(max_check_rows, h)):
        row = frame[i, :, :]
        if np.std(row) < std_threshold:
            gray_bar_height += 1
        else:
            break
    if gray_bar_height > 0:
        return frame[gray_bar_height:, :], gray_bar_height
    return frame, 0

def add_cctv_overlay(frame, frame_counter):
    """Add a CCTV-style overlay with timestamp, REC indicator, and noise."""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Timestamp in top-left corner
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(overlay, timestamp, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # "REC" indicator in top-right corner (blinking every 30 frames)
    if (frame_counter // 30) % 2 == 0:
        cv2.putText(overlay, "REC", (width - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (width - 30, 30), 8, (0, 0, 255), -1)
    
    # CCTV border
    border_thickness = 10
    cv2.rectangle(overlay, (border_thickness, border_thickness),
                  (width - border_thickness, height - border_thickness),
                  (255, 255, 255), 2)
    
    overlay_lines = overlay.copy()

    # Draw the scan lines in light gray
    '''for y in range(0, height, 20):
        cv2.line(overlay_lines, (0, y), (width, y), (200, 200, 200), 1)  # Light gray color
    '''
    # Blend the overlay with the original frame (adjust opacity by changing alpha)
    alpha = 0.3  # Opacity level (0 = fully transparent, 1 = fully visible)
    cv2.addWeighted(overlay_lines, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Grainy noise overlay
    noise = np.random.randint(0, 50, frame.shape, dtype='uint8')
    overlay = cv2.addWeighted(overlay, 0.98, noise, 0.02, 0)
    
    return overlay

def add_cctv_overlay0(frame, frame_counter):
    """Add a CCTV-style overlay with timestamp, REC indicator, and noise."""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    
    
    overlay_lines = overlay.copy()

    # Draw the scan lines in light gray
    for y in range(0, height, 20):
        cv2.line(overlay_lines, (0, y), (width, y), (200, 200, 200), 1)  # Light gray color
    
    # Blend the overlay with the original frame (adjust opacity by changing alpha)
    alpha = 0.3  # Opacity level (0 = fully transparent, 1 = fully visible)
    cv2.addWeighted(overlay_lines, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Grainy noise overlay
    noise = np.random.randint(0, 50, frame.shape, dtype='uint8')
    overlay = cv2.addWeighted(overlay, 0.98, noise, 0.02, 0)
    
    return overlay

def add_logo_overlay(frame, logo_path):
    """Add a logo overlay on the frame."""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Load the logo image
    logo = cv2.imread('media/logo_clear.png', cv2.IMREAD_UNCHANGED)  # Load with transparency if possible

    print(logo)
    
    # Resize the logo if necessary
    logo_width = 300  # Desired logo width
    aspect_ratio = logo.shape[1] / logo.shape[0]
    logo_height = int(logo_width / aspect_ratio)
    logo_resized = cv2.resize(logo, (logo_width, logo_height))
    
    # Get the region of interest (ROI) for the logo's position (e.g., top-right corner)
    roi_x = 20  # 10 pixels margin from the right edge
    roi_y = height - 83  # 10 pixels margin from the top edge
    
    # Get the logo's alpha channel if it exists (for transparency)
    if logo_resized.shape[2] == 4:
        alpha_channel = logo_resized[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        logo_resized = logo_resized[:, :, :3]  # Remove alpha from the logo
    else:
        alpha_channel = np.ones((logo_resized.shape[0], logo_resized.shape[1]))  # Fully opaque logo

    # Create an overlay of the logo on the frame
    for c in range(0, 3):
        overlay[roi_y:roi_y+logo_resized.shape[0], roi_x:roi_x+logo_resized.shape[1], c] = (
            alpha_channel * logo_resized[:, :, c] + (1 - alpha_channel) * overlay[roi_y:roi_y+logo_resized.shape[0], roi_x:roi_x+logo_resized.shape[1], c]
        )

    return overlay

# -------------------------------------------------
# 5) Main Loop
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_counter += 1
    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Dynamic Gray Bar Removal
    frame, removed_height = remove_gray_bar(frame, max_check_rows=50, std_threshold=10)

    # Update dimensions after crop
    orig_h, orig_w = frame.shape[:2]

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    

    # Create a transparent background with an alpha channel (RGBA)
    black_background = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)  # Transparent

    # Face Detection and Tracking
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * orig_w)
            y_min = int(bbox.ymin * orig_h)
            box_w = int(bbox.width * orig_w)
            box_h = int(bbox.height * orig_h)

            # Clamp the bounding box inside the frame
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_w, x_min + box_w)
            y_max = min(orig_h, y_min + box_h)

            # Center of bounding box
            cx = x_min + box_w // 2
            cy = y_min + box_h // 2

            # Match or create a tracked face
            assigned_color = None
            for tracked in tracked_faces:
                prev_cx, prev_cy = tracked['center']
                if math.hypot(cx - prev_cx, cy - prev_cy) < CENTER_DISTANCE_THRESHOLD:
                    assigned_color = tracked['color']
                    tracked['center'] = (cx, cy)
                    tracked['last_seen'] = frame_counter
                    break

            if assigned_color is None:
                used_colors = {f['color'] for f in tracked_faces}
                assigned_color = get_unique_color(used_colors)
                tracked_faces.append({
                    'center': (cx, cy),
                    'color': assigned_color,
                    'last_seen': frame_counter
                })

            # Draw rectangle and apply pixelation to the face region on the transparent background
            cv2.rectangle(black_background, (x_min, y_min), (x_max, y_max), assigned_color, -1)  # Filled rectangle with alpha
            cv2.rectangle(black_background, (x_min + 20, y_min + 20), (x_max + 20, y_max + 20), assigned_color, 5)  # Filled rectangle with alpha

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), assigned_color, 2)
            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size != 0:
                # Reduce the size more for stronger pixelation (e.g., 8x8 instead of 16x16)
                small = cv2.resize(face_roi, (12, 12), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                frame[y_min:y_max, x_min:x_max] = pixelated
            

            # Confidence text
            confidence = detection.score[0] if detection.score else 0
            confidence_text = f"Face: {confidence * 100:.1f}%"
            text_x = x_min
            text_y = y_min - 10 if (y_min - 10) > 0 else y_min + 20
            cv2.putText(black_background, confidence_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, assigned_color, 2, cv2.LINE_AA)

    # Remove stale tracked faces
    tracked_faces = [
        f for f in tracked_faces
        if (frame_counter - f['last_seen']) <= MAX_MISSING_FRAMES
    ]

    # Apply CCTV overlay to the original frame
        

    frame_with_overlay = add_cctv_overlay0(frame, frame_counter)

    # ---------------------------------------------------------
    # "Cover" approach: Scale the frame to completely fill the screen,
    # cropping out any parts that exceed the screen dimensions.
    # ---------------------------------------------------------
    if screen_w > 0 and screen_h > 0:
        frame_aspect = orig_w / orig_h
        screen_aspect = screen_w / screen_h

        if frame_aspect > screen_aspect:
            new_h = screen_h
            new_w = int(new_h * frame_aspect)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            x_offset = (new_w - screen_w) // 2
            display_frame = resized_frame[:, x_offset:x_offset+screen_w]
        else:
            new_w = screen_w
            new_h = int(new_w / frame_aspect)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y_offset = (new_h - screen_h) // 2
            display_frame = resized_frame[y_offset:y_offset+screen_h, :]
    else:
        display_frame = frame_with_overlay

    # Convert the frame to grayscale for display
    gray_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale frame back to a 3-channel frame for blending
    gray_frame_colored = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Blend the three frames (original, grayscale, and transparent background with squares)
    combined_frame = cv2.addWeighted(display_frame, 0, gray_frame_colored, 1, 0)
    combined_frame = cv2.addWeighted(combined_frame, 0.5, black_background[:, :, :3], 0.5, 0)

    frame_with_overlay = add_cctv_overlay(combined_frame, frame_counter)

    finalFrame = add_logo_overlay(frame_with_overlay, frame_counter)
    cv2.imshow(window_name, finalFrame)



    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()