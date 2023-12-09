from transformers import AutoProcessor, BarkModel
import streamlit as st
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model():
    model = BarkModel.from_pretrained("suno/bark-small")
    model = model.to(device)
    return model

def load_processor():
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    return processor

def preprocess_text(text):
    processor = load_processor()
    output = processor(text)
    return output

def generate_tts(text):
    inputs = preprocess_text(text)
    model = load_model()

    with st.spinner('Генерируется аудио...'):
        speech_output = model.generate(**inputs.to(device))

        rate = model.generation_config.sample_rate
        audio = speech_output[0].cpu().numpy()

        return [audio, rate]
    
st.title('Text-to-Speech с аннотациями')
text = st.text_area('Введите текст')
result = st.button('Произнести')
if result and text:
    audio, rate = generate_tts(text)
    st.audio(audio, sample_rate=rate)