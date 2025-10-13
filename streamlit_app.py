import streamlit as st
import tempfile
import os
import requests
import json
import re
import unicodedata
import time
import httpx
import torch
from pyannote.audio import Pipeline
import whisper
from anthropic import Anthropic

# === Config ===
ASSEMBLYAI_API_KEY = st.secrets["api_keys"]["assemblyai"]
HUGGINGFACE_TOKEN = st.secrets["api_keys"]["huggingface"]
CLAUDE_API_KEY = st.secrets["api_keys"]["anthropic"]

client = Anthropic(api_key=CLAUDE_API_KEY)
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# === Streamlit UI ===
st.title("🎙️ Audio Transcription + Speaker Diarization")
st.caption("Choose transcription engine and speaker detection method.")

engine_choice = st.radio(
    "Select engine:",
    ["AssemblyAI (Transcription + Diarization)", "Whisper + Pyannote"]
)

uploaded_file = st.file_uploader("Upload a WAV/MP3/M4A file", type=["wav", "mp3", "m4a"])

# === Helper functions ===
def robust_sanitize_multilingual_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    text = text.replace("�", "")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", " ", text)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_with_claude(text):
    sanitized_text = robust_sanitize_multilingual_text(text)
    prompt = f"""Analyze the sentiment and content of the following text and respond ONLY with valid JSON.

    Text to analyze:
    {sanitized_text}

    Requirements:
    1. Determine the overall sentiment (Positive, Negative, or Neutral)
    2. Assign a sentiment score from 0.0 (most negative) to 1.0 (most positive), where 0.5 is neutral
    3. Categorize the main topic/theme
    4. Explain why you assigned this sentiment
    5. Provide a brief summary

    Respond with this exact JSON structure (no additional text):
    {{
    "Category": "the main topic or theme",
    "Sentiment Score": 0.0,
    "Sentiment Label": "Positive/Negative/Neutral",
    "Reason": "explanation of sentiment determination",
    "Summary": "brief summary of the text"
    }}
    Examples:
        - "I love this product!" → Score: 0.9, Label: "Positive"
        - "This is terrible and broken" → Score: 0.1, Label: "Negative"  
        - "The meeting is scheduled for 3pm" → Score: 0.5, Label: "Neutral"
    Return ONLY the JSON object, no markdown formatting or extra text."""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text.strip()
        response_text = re.sub(r"^```(json)?", "", response_text)
        response_text = re.sub(r"```$", "", response_text).strip()
        match = re.search(r"\{[\s\S]*\}", response_text)
        json_str = match.group(0) if match else response_text
        result = json.loads(json_str)
        # Ensure required keys
        for key in ["Category", "Sentiment Score", "Sentiment Label", "Reason", "Summary"]:
            if key not in result:
                result[key] = "Unknown" if key != "Sentiment Score" else 0.5
        result["Sentiment Score"] = float(result["Sentiment Score"])
        result["Sentiment Score"] = max(0.0, min(1.0, result["Sentiment Score"]))
        if result["Sentiment Label"] not in ["Positive", "Negative", "Neutral"]:
            result["Sentiment Label"] = "Neutral"
        return result
    except Exception as e:
        st.error(f"Claude API error: {e}")
        return {
            "Category": "API Error",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Reason": f"API call failed: {str(e)}",
            "Summary": "Unable to analyze text",
        }

# === AssemblyAI helpers ===
def upload_file_assembly(file_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(file_path, "rb") as f:
        with httpx.Client(http2=False, timeout=None) as client:
            response = client.post(UPLOAD_URL, headers=headers, content=f)
            response.raise_for_status()
            return response.json()["upload_url"]

def transcribe_assemblyai(audio_url, speaker_labels=False):
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    data = {"audio_url": audio_url, "speaker_labels": speaker_labels, "language_detection": True}
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
        time.sleep(3)

# === Whisper + Pyannote helpers ===
@st.cache_resource
def load_diarization_model():
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline

def diarize_with_pyannote(audio_path):
    pipeline = load_diarization_model()
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": round(turn.start,2), "end": round(turn.end,2), "speaker": f"SPEAKER_{speaker}"})
    return segments

def normalize_speakers(segments):
    speaker_map = {}
    next_label = ord("A")
    normalized = []
    for seg in segments:
        orig_speaker = seg["speaker"]
        if orig_speaker not in speaker_map:
            speaker_map[orig_speaker] = f"Speaker {chr(next_label)}"
            next_label += 1
        normalized.append({
            "speaker": speaker_map[orig_speaker],
            "text": seg["text"],
            "start": seg.get("start"),
            "end": seg.get("end")
        })
    return normalized

# === File processing ===
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        path = tmp.name

    try:
        if engine_choice.startswith("AssemblyAI"):
            audio_url = upload_file_assembly(path)
            transcript_result = transcribe_assemblyai(audio_url, speaker_labels=True)
            full_text = transcript_result.get("text", "")
            # Extract speaker segments from AssemblyAI
            segments = []
            words = transcript_result.get("words", [])
            if words:
                current_speaker = None
                current_text = []
                for w in words:
                    speaker = w.get("speaker")
                    if speaker != current_speaker:
                        if current_speaker:
                            segments.append({
                                "speaker": current_speaker,
                                "text": " ".join(current_text)
                            })
                        current_speaker = speaker
                        current_text = [w["text"]]
                    else:
                        current_text.append(w["text"])
                if current_text:
                    segments.append({"speaker": current_speaker, "text": " ".join(current_text)})
            speakers = normalize_speakers(segments)

        else:  # Whisper + Pyannote
            st.info("Transcribing with Whisper...")
            whisper_model = whisper.load_model("large")
            result = whisper_model.transcribe(path, task='transcribe', language="hi")
            full_text = result["text"]
            print("full trxt",full_text)
            st.info("Diarizing with Pyannote...")
            speaker_segments = diarize_with_pyannote(path)

            # Align text with speakers (simple split by sentences)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_text) if s.strip()]
            segments = []
            for i, sent in enumerate(sentences):
                seg = speaker_segments[i % len(speaker_segments)]
                seg["text"] = sent
                segments.append(seg)
            speakers = normalize_speakers(segments)

        # Display transcription
        st.subheader("📄 Full Transcription")
        st.text_area("Complete Text", full_text, height=200)

        # Display speaker conversation
        st.subheader("👥 Speaker-wise Conversation")
        for utt in speakers:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"**{utt['speaker']}**")
            with col2:
                st.write(utt['text'])
            st.divider()

        # Claude analysis
        if full_text:
            st.subheader("🤖 Analysis: Summary & Sentiment")
            with st.spinner("Analyzing with Claude..."):
                analysis = analyze_with_claude(full_text)

            st.markdown(f"""
        **Category:** {analysis.get("Category", "N/A")}

        **Sentiment Score:** {analysis.get("Sentiment Score", 0.5):.2f}  
        **Sentiment Label:** {analysis.get("Sentiment Label", "Neutral")}

        **Reason:** {analysis.get("Reason", "N/A")}

        **Summary**  
        {analysis.get("Summary", "No summary available")}
        """)

    except Exception as e:
        st.error(f"Error processing audio: {e}")
    finally:
        if os.path.exists(path):
            os.unlink(path)
