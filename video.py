import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("D:/Python/FireDetector/project_combineData/combineData_16_epo3/weights/best.pt")
# Image selection
video = None
video_source = st.radio("Select video source:", ("Enter URL", "Upload from Computer"))

if video_source == "Upload from Computer":
        # File uploader for video
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_file is not None:
            video_bytes = uploaded_file.read()
            st.video('video_bytes')


            cap = cv2.VideoCapture(uploaded_file)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
        else:
            image = None

            #cv2.imshow("Capturing for facial recog", frame)
           # media_predictions = model.predict(source=uploaded_file, show=True)


