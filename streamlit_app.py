import streamlit as st
import requests
import tempfile

ASSEMBLYAI_API_KEY = "4600824a76e84ba5948711363fb84158"
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

def upload_file(file_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(file_path, "rb") as f:
        r = requests.post(UPLOAD_URL, headers=headers, data=f)
    r.raise_for_status()
    return r.json()["upload_url"]

def transcribe(audio_url):
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    data = {"audio_url": audio_url, "speaker_labels": True, "language_detection": True}
    r = requests.post(TRANSCRIPT_URL, headers=headers, json=data)
    r.raise_for_status()
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
        st.info("Transcription in progress...")
        import time; time.sleep(3)

def normalize_speakers(utterances):
    """Rename speakers to Speaker A, B, ... and remove timestamps"""
    speaker_map = {}
    next_label = ord("A")
    normalized = []
    for utt in utterances:
        orig_speaker = utt["speaker"]
        if orig_speaker not in speaker_map:
            speaker_map[orig_speaker] = f"Speaker {chr(next_label)}"
            next_label += 1
        normalized.append({
            "speaker": speaker_map[orig_speaker],
            "text": utt["text"]
        })
    return normalized

# Streamlit UI
st.title("Audio Transcription + Multi-Speaker Diarization")
uploaded_file = st.file_uploader("Upload a WAV/MP3", type=["wav","mp3","m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        path = tmp.name
    
    audio_url = upload_file(path)
    result = transcribe(audio_url)
    
    st.subheader("Full Transcription")
    st.text_area("Text", result.get("text", ""), height=300)

    st.subheader("Speakers")
    speakers = normalize_speakers(result.get("utterances", []))
    for utt in speakers:
        st.write(f"{utt['speaker']}: {utt['text']}")

