import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
options='Video'
# Video
if options == 'Video':
    upload_video_file = st.sidebar.file_uploader(
        'Upload Video', type=['mp4', 'avi', 'mkv'])
    if upload_video_file is not None:
        pred = st.checkbox(f'Predict Using {model_type}')

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        # if pred: