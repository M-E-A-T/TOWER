import cv2
import mediapipe as mp
import math
import random
import numpy as np

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

def get_unique_color(used_colors):
    """Generate a random color (BGR) not in used_colors."""
    while True:
        new_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        if new_color not in used_colors:
            return new_color

def remove_gray_bar(frame, max_check_rows=50, std_threshold=10):
    """
    Examine the top rows of the frame and remove any that appear uniformly gray.
    :param frame: The input image.
    :param max_check_rows: Maximum number of top rows to check.
    :param std_threshold: If the standard deviation of a row's pixels is below this,
                          consider it part of the gray bar.
    :return: The cropped frame.
    """
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

            # Draw rectangle and apply pixelation to the face region
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), assigned_color, 2)
            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size != 0:
                small = cv2.resize(face_roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                frame[y_min:y_max, x_min:x_max] = pixelated

            # Confidence text
            confidence = detection.score[0] if detection.score else 0
            confidence_text = f"Face: {confidence * 100:.1f}%"
            text_x = x_min
            text_y = y_min - 10 if (y_min - 10) > 0 else y_min + 20
            cv2.putText(frame, confidence_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, assigned_color, 2, cv2.LINE_AA)

    # Remove stale tracked faces
    tracked_faces = [
        f for f in tracked_faces
        if (frame_counter - f['last_seen']) <= MAX_MISSING_FRAMES
    ]

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
        display_frame = frame

    # Use display_frame directly (no feedback effect)
    combined_frame = display_frame

    # ---------------------------------------------------------
    # Saturate blue hues and add enhanced artistic noise
    # ---------------------------------------------------------
    # Convert to HSV to work in the color space
    hsv = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2HSV)

    # Define the blue hue range (tweak as needed)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Increase saturation for blue pixels
    saturation_factor = 1.5  # Increase saturation by 50%
    hsv[:, :, 1] = np.where(blue_mask > 0,
                             np.clip(hsv[:, :, 1].astype(np.float32) * saturation_factor, 0, 255),
                             hsv[:, :, 1])
    hsv = hsv.astype(np.uint8)
    
    # Convert back to BGR for noise addition
    modified_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Add enhanced artistic Gaussian noise to blue pixels only
    noise_strength = 50  # Standard deviation for Gaussian noise
    noise = np.random.normal(0, noise_strength, modified_frame.shape).astype(np.int16)
    
    modified_frame = modified_frame.astype(np.int16)
    blue_pixels = blue_mask > 0  # Boolean mask for blue pixels
    for c in range(3):
        channel = modified_frame[:, :, c]
        channel[blue_pixels] += noise[:, :, c][blue_pixels]
        channel[blue_pixels] = np.clip(channel[blue_pixels], 0, 255)
        modified_frame[:, :, c] = channel
    modified_frame = modified_frame.astype(np.uint8)
    
    combined_frame = modified_frame

    cv2.imshow(window_name, combined_frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
