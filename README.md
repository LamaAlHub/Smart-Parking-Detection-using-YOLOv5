# Smart Parking Detection using YOLOv5
 
A computer vision system that detects parking spaces in images and video and tells you how many are **empty** and how many are **occupied**. It uses a custom-trained YOLOv5 model and comes with a simple Streamlit web app.
 
## Features
 
- Detects parking spaces and classifies each one as `empty` or `occupied`
- Counts empty and occupied spaces and shows the totals on the output
- Streamlit web app: upload a parking lot image and get the annotated result
- Notebook for running the model on a single image or on a video stream (live counts per frame)
## Model
 
| | |
|---|---|
| Architecture | YOLOv5s (custom-trained) |
| Classes | `0: empty`, `1: occupied` |
| Size | ~7M parameters, 15.8 GFLOPs |
| Weights | `best.pt` (included in this repo) |
 
## Project structure
 
```
├── app.py              # Streamlit web app (image upload → prediction)
├── modelrunCode.ipynb  # Run the model on an image or a video with OpenCV
├── best.pt             # Trained YOLOv5 weights
├── requirements.txt    # Python dependencies
└── README.md
```
 
## Installation
 
```bash
git clone https://github.com/LamaAlHub/Smart-Parking-Detection-using-YOLOv5.git
cd Smart-Parking-Detection-using-YOLOv5
pip install -r requirements.txt
pip install streamlit
```
 
The model is loaded through `torch.hub`, so the first run needs an internet connection to download the YOLOv5 code.
 
## Usage
 
### 1. Set the model path
 
In `app.py` and `modelrunCode.ipynb`, the model path points to a local Windows folder. Change it to the `best.pt` file in this repo:
 
```python
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
```
 
### 2. Run the web app
 
```bash
streamlit run app.py
```
 
Open the link Streamlit prints (usually http://localhost:8501), upload a parking lot image, and click **Predict**. The app shows the annotated image with the number of empty and occupied spaces.
 
### 3. Run on an image or a video
 
Open `modelrunCode.ipynb` in Jupyter or VS Code, set the path to your image or video, and run the cells. A window opens with the detections and live counts. Press **q** to close the video window.
 
## How it works
 
1. The image or video frame is passed to the YOLOv5 model.
2. The model returns a bounding box and a class (`empty` or `occupied`) for each parking space.
3. Detections below the confidence threshold (`model.conf`, 0.5 for images and 0.6 for video) are dropped.
4. The remaining detections are counted per class and drawn on the image.
## Notes
 
- The code includes a `pathlib.PosixPath = pathlib.WindowsPath` line. It is needed on **Windows** when the model was trained on Linux or Colab. Remove it if you run the project on Linux or macOS, or it will raise an error.
- `force_reload=True` downloads YOLOv5 again on every run. You can remove it after the first run to start faster.
## Tech stack
 
Python · PyTorch · YOLOv5 · OpenCV · Streamlit · NumPy · Pillow
