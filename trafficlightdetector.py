import cv2
import numpy as np

# HSV color ranges for Red (two ranges), Yellow, and Green
COLOR_RANGES = {
    'Red1':   [(0, 100, 100), (10, 255, 255)],
    'Red2':   [(170, 100, 100), (180, 255, 255)],  
    'Yellow': [(20, 100, 100), (30, 255, 255)],
    'Green':  [(40, 100, 100), (70, 255, 255)]
}

# Parameter to ignore the tiny blobs
MIN_AREA = 300
# Parameter 0=flat and 1=perfect circle
CIRCULARITY = 0.5    

def detect_color_objects(hsv_frame):
    detections = []

    for color_name, (lower, upper) in COLOR_RANGES.items():
        # Create mask
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))

        # Noise cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            # To check the circularity (4πA / P²) 
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < CIRCULARITY:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Merges Red1 + Red2 into single "Red"
            cname = "Red" if "Red" in color_name else color_name
            detections.append((cname, x, y, w, h))

    return detections

def classify_state(detections):
    """Decide the traffic light state based on detections (priority: Red > Yellow > Green)."""
    colors = [d[0] for d in detections]
    if "Red" in colors: return "Red"
    if "Yellow" in colors: return "Yellow"
    if "Green" in colors: return "Green"
    return "None"

def draw_detections(frame, detections, state):
    for color_name, x, y, w, h in detections:
        color = (0,0,255) if color_name=="Red" else (0,255,255) if color_name=="Yellow" else (0,255,0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"STATE: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)  #if you need to upload then put the path of the video in mp4 or jpg format for image in place of zero

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        blurred = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Detects traffic light colors
        detections = detect_color_objects(hsv)
        state = classify_state(detections)
        draw_detections(frame, detections, state)
        cv2.imshow("Traffic Light Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




