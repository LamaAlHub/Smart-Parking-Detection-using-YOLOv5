import torch 
import cv2
import pathlib
import numpy as np
import streamlit as st 
from  ultralytics import YOLO
from PIL import Image
temp = pathlib.PosixPath 
pathlib.PosixPath = pathlib.WindowsPath


st.title("Car Space YOLO")

model = torch.hub.load("ultralytics/yolov5","custom",path = r"C:\Users\imyla\Downloads\CarSpaceYOLO\yolov5\runs\train\exp\weights\best.pt",force_reload = True)

uploadded_file = st.file_uploader("upload image",type = ["jpg","png","jpeg"])
if uploadded_file:
    image = Image.open(uploadded_file)
    st.image(image, caption= "Original Image")
    
    if st.button("Predict"):
        model.conf = 0.5
        result = model(image)
        
        emptySpaceCount = 0
        occupiedSpaceCount =0
        
        for detection in result.xyxy[0]:
            class_id = int(detection[5])
            class_name = model.names[class_id]
            if class_name == "empty":
              emptySpaceCount += 1
            elif class_name == "occupied":
                occupiedSpaceCount += 1
            
        annotated_img = np.squeeze(result.render())
        st.image(annotated_img, caption = "Annotated Image")
        st.header("Parking Space Analysis")
        st.text(f"Empty Spaces: {emptySpaceCount}")
        st.text(f"Occupied Spaces: {occupiedSpaceCount}")
