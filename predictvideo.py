
import os
import torch
from ultralytics import YOLO
model = YOLO("D:/Python/FireDetector/project_combineData/combineData_16_epo3/weights/best.pt")

#For video
media_predictions = model.predict(source=r"D:/Python/FireDetector/examples/video4.mp4", show=True)
#media_predictions.show()
#media_predictions.save("output_video.mp4")
