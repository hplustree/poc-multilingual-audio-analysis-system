# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/analyze"  # your FastAPI endpoint

st.title("Audio Transcription + Multi-Speaker Diarization")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.info("Uploading and transcribing audio, please wait... ⏳")
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    
    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Transcription completed!")

            st.subheader("Detected Language")
            st.write(result.get("language", "Unknown"))

            st.subheader("Full Transcription")
            st.text_area("Transcription", value=result.get("transcription", ""), height=300)

            st.subheader("Speakers")
            for speaker in result.get("speakers", []):
                st.write(f"{speaker['speaker']} [{speaker['start']} - {speaker['end']}]: {speaker['text']}")

        else:
            st.error(f"Transcription failed: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
