import streamlit as st
import torch
import os
import torchaudio
import torchaudio.transforms as T
from transformers import pipeline, Wav2Vec2ForCTC, AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor
import subprocess
from streamlit_mic_recorder import mic_recorder  # Import new audio recorder

# Set page title and styling
st.set_page_config(page_title="🎤 Dzongkha ASR", page_icon="🔊", layout="centered")
st.markdown("""
    <style>
        .big-font { font-size:22px !important; font-weight: bold; }
        .highlight { color: #007BFF; font-weight: bold; }
        
        /* Transcription Output Box */
        .transcription-box { 
            font-size: 28px; 
            color: #333; 
            font-weight: bold; 
            text-align: center; 
            padding: 20px; 
            background-color: white; 
            border-radius: 10px; 
            border: 2px solid #007BFF; 
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Custom Button */
        div[data-testid="stButton"] button {
            background-color: #007BFF !important; 
            color: white !important; 
            font-size: 18px !important; 
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Ensure FFmpeg is installed
# This code checks if **FFmpeg** is installed. If not, it shows an error message in a **Streamlit** app and stops execution.
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except FileNotFoundError:
        return False

if not check_ffmpeg():
    st.error("⚠️ FFmpeg is not installed. Please install FFmpeg to continue.")
    st.stop()

# Load ASR Model Components
model_id = "./checkpoint-5166"

# Load model
model = Wav2Vec2ForCTC.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
processor = Wav2Vec2Processor(feature_extractor, tokenizer)

st.success("✅ Model Loaded Successfully")

# Configure for Dzongkha
# This sets the model's language to **Dzongkha** and the task to **transcription**, if those settings exist. 
if hasattr(model.config, 'language'):
    model.config.language = "dzo"
if hasattr(model.config, 'task'):
    model.config.task = "transcribe"

# Load Pipeline
# This creates a speech-to-text **ASR pipeline**, using the specified model, tokenizer, and feature extractor. It runs on **GPU if available**, otherwise on **CPU**. 
classifier = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    device=0 if torch.cuda.is_available() else -1
)

# Function to resample audio to 16kHz
def resample_audio(waveform, original_sample_rate, target_sample_rate=16000):
    if original_sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, target_sample_rate

# Convert non-standard WAV to PCM WAV
# This converts an audio file to **PCM WAV (16-bit, 16kHz)** using **FFmpeg**. If conversion fails, it shows an error in **Streamlit**. 
def convert_to_pcm_wav(input_path, output_path):
    try:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-acodec", "pcm_s16le", "-ar", "16000", output_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"⚠️ Error converting audio file: {e}")
        return None

# Streamlit UI Setup
st.markdown("""
    <div style="text-align: center; margin-bottom: 10px;">
        <h1>🎤 Dzongkha Automatic Speech Recognition</h1>
        <p style="font-size:22px; font-weight: bold;">Transcribe Dzongkha speech to text seamlessly</p>
    </div>
    <br>
""", unsafe_allow_html=True)

# Add minimal spacing before "Record Audio"
st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    st.subheader("🎙️ Record Audio")

    st.write("Click the button below to start recording:")
    audio_data = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="🛑 Stop Recording")

    st.subheader("📂 Upload Audio File")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "ogg", "mp3"])

audio_path = None

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

elif audio_data is not None:
    st.audio(audio_data["bytes"], format="audio/wav")
    raw_audio_path = "recorded_audio.raw.wav"
    audio_path = "recorded_audio.wav"
    with open(raw_audio_path, "wb") as f:
        f.write(audio_data["bytes"])
    audio_path = convert_to_pcm_wav(raw_audio_path, audio_path)

if audio_path and audio_path is not None:
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        st.write(f"📢 **Original Sample Rate:** `{sample_rate} Hz`")

        # Resample to 16kHz
        waveform, new_sample_rate = resample_audio(waveform, sample_rate, 16000)
        st.write(f"✅ **Resampled to:** `{new_sample_rate} Hz`")

        # Convert to model's expected format
        inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=new_sample_rate, return_tensors="pt")

        # Perform Speech Recognition
        with st.spinner("Transcribing..."):
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)[0]

        # Display Transcription in a Box
        st.subheader("📜 Transcription:")
        st.markdown(f"<div class='transcription-box'>{transcription}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error processing the audio file: {str(e)}")

    # Clean up temp audio file
    os.remove(audio_path)
