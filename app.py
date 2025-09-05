import cv2
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile

# Define HSV ranges
COLOR_RANGES = {
    'Red1':   [(0, 100, 100), (10, 255, 255)],
    'Red2':   [(170, 100, 100), (180, 255, 255)],
    'Yellow': [(20, 100, 100), (30, 255, 255)],
    'Green':  [(40, 100, 100), (70, 255, 255)]
}

MIN_AREA = 300
CIRCULARITY = 0.5

def detect_color_objects(hsv_frame):
    detections = []
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA: continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < CIRCULARITY: continue
            x, y, w, h = cv2.boundingRect(cnt)
            cname = "Red" if "Red" in color_name else color_name
            detections.append((cname, x, y, w, h))
    return detections

def classify_state(detections):
    colors = [d[0] for d in detections]
    if "Red" in colors: return "Red"
    if "Yellow" in colors: return "Yellow"
    if "Green" in colors: return "Green"
    return "None"

def process_video(video_path):
    cap = cv2.VideoCapture(video_path) #add the path of the video or put 0 for the web cam to work
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = detect_color_objects(hsv)
        state = classify_state(detections)
        for color_name, x, y, w, h in detections:
            color = (0,0,255) if color_name=="Red" else (0,255,255) if color_name=="Yellow" else (0,255,0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frames.append(frame)
    cap.release()
    return frames

# Streamlit UI
st.title("Traffic Light Detection System")
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_file.read())
        video_path = temp.name

    st.info("Processing video... please wait ")
    frames = process_video(video_path)

    st.success("Done! Showing first few frames:")
    for f in frames[:30]:  # Show only first 30 frames to save resources
        st.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

