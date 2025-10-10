import time
import streamlit as st
import tempfile
import subprocess
import requests

ASSEMBLYAI_API_KEY = "4600824a76e84ba5948711363fb84158"
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

def convert_audio(input_path, output_path):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path]
    subprocess.run(cmd, check=True)

def upload_file(file_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(file_path, "rb") as f:
        r = requests.post(UPLOAD_URL, headers=headers, data=f)
    return r.json()["upload_url"]

def transcribe(audio_url):
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    data = {"audio_url": audio_url, "speaker_labels": True, "language_detection": True}
    r = requests.post(TRANSCRIPT_URL, headers=headers, json=data)
    transcript_id = r.json()["id"]
    
    polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
    while True:
        r = requests.get(polling_url, headers=headers)
        res = r.json()
        if res["status"] == "completed":
            return res
        elif res["status"] == "error":
            st.error(res.get("error", "Unknown error"))
            return {}
        time.sleep(3)

# Streamlit UI
st.title("Audio Transcription + Multi-Speaker Diarization")
uploaded_file = st.file_uploader("Upload a WAV/MP3", type=["wav","mp3"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        path = tmp.name
    
    converted_path = path.replace(".wav","_converted.wav")
    convert_audio(path, converted_path)
    audio_url = upload_file(converted_path)
    result = transcribe(audio_url)
    
    st.subheader("Transcription")
    st.text_area("Full Text", result.get("text", ""), height=300)
    st.subheader("Speakers")
    for utt in result.get("utterances", []):
        st.write(f"{utt['speaker']} [{utt['start']}-{utt['end']}]: {utt['text']}")
