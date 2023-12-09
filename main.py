from transformers import VitsTokenizer, VitsModel
import streamlit as st
import torch

MODEL_ID = "facebook/mms-tts-eng"


@st.cache_resource
def load_model():
    model = VitsModel.from_pretrained(MODEL_ID)
    return model

@st.cache_resource
def load_processor():
    processor = VitsTokenizer.from_pretrained(MODEL_ID)
    return processor

def preprocess_text(text):
    processor = load_processor()
    output = processor(text=text, return_tensors="pt")
    return output

def generate_tts(text):
    inputs = preprocess_text(text)
    model = load_model()

    with st.spinner('Генерируется аудио...'):
        with torch.no_grad():
            outputs = model(**inputs)

        waveform = outputs.waveform[0]
        return waveform, model.config.sampling_rate


st.title('Text-to-Speech 🇺🇸')
text = st.text_area('Введите текст')
result = st.button('Произнести')
if result and text:
    audio, rate = generate_tts(text)
    st.audio(audio, sample_rate=rate)