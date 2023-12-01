import torch
import streamlit as st
from ultralyticsplus import YOLO, render_result
from PIL import Image

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NMS_CONF = 0.25
IOU = 0.45
NMS_AGNOSTIC = False
MAX_DETECTIONS = 1000

# load model
@st.cache_resource()
def load_model():
    model = YOLO('keremberke/yolov8s-table-extraction')
    model.overrides['conf'] = NMS_CONF
    model.overrides['iou'] = IOU
    model.overrides['agnostic_nms'] = NMS_AGNOSTIC
    model.overrides['max_det'] = MAX_DETECTIONS
    model.to(DEVICE)
    return model

def find_tables(img):
    model = load_model()
    results = model.predict(img)
    render = render_result(model=model, image=img, result=results[0])
    return render

st.title('Найти таблицы на картинке')
image_buffer = st.file_uploader('Картинка с таблицей', type=['png', 'jpg'])
result = st.button('Найти')
if result and image_buffer:
    image = Image.open(image_buffer).convert('RGB')
    predictions = find_tables(image)
    st.image(predictions)
