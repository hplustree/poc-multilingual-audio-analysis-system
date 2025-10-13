import streamlit as st
import requests
import tempfile
import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import time
import json
import re
import unicodedata
import httpx
import os

# === Config ===
ASSEMBLYAI_API_KEY = st.secrets["api_keys"]["assemblyai"]
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

CLAUDE_API_KEY = st.secrets["api_keys"]["anthropic"]
client = Anthropic(api_key=CLAUDE_API_KEY)
# === Helper functions ===
def upload_file(file_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(file_path, "rb") as f:
        # Upload in streaming chunks (recommended by AssemblyAI)
        with httpx.Client(http2=False, timeout=None) as client:
            response = client.post(
                UPLOAD_URL,
                headers=headers,
                content=f  # ✅ raw bytes, not multipart
            )
            response.raise_for_status()
            return response.json()["upload_url"]

def transcribe(audio_url):
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    data = {"audio_url": audio_url, "speaker_labels": True, "language_detection": True}
    r = requests.post(TRANSCRIPT_URL, headers=headers, json=data)
    r.raise_for_status()
    transcript_id = r.json()["id"]

    polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
    with st.spinner():
        while True:
            r = requests.get(polling_url, headers=headers)
            res = r.json()
            if res["status"] == "completed":
                return res
            elif res["status"] == "error":
                st.error(res.get("error", "Unknown error"))
                return {}
            time.sleep(3)

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


def robust_sanitize_multilingual_text(raw_text: str) -> str:
    """
    Fixes encoding issues, removes invalid characters, and normalizes Unicode.
    Works for Tamil, Malayalam, Hindi, English mixed text.
    """
    if not raw_text:
        return ""
    
    # Replace invalid bytes with placeholder
    text = raw_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    
    # Remove non-printable control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    
    # Remove replacement symbols
    text = text.replace("�", "")
    
    # Normalize Unicode to NFC form
    text = unicodedata.normalize("NFC", text)
    
    # Remove invisible zero-width chars
    text = re.sub(r"[\u200B-\u200D\uFEFF]", " ", text)
    
    # Escape special characters that might break JSON
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    
    # Collapse multiple spaces
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
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text.strip()  
        # Remove Markdown formatting if present
        response_text = re.sub(r"^```(json)?", "", response_text)
        response_text = re.sub(r"```$", "", response_text).strip()
        
        # Extract JSON object
        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            json_str = match.group(0)
        else:
            json_str = response_text
        
        # Parse JSON
        result = json.loads(json_str)
        
        # Validate required keys
        required_keys = ["Category", "Sentiment Score", "Sentiment Label", "Reason", "Summary"]
        for key in required_keys:
            if key not in result:
                result[key] = "Unknown" if key != "Sentiment Score" else 0.5
        
        # Ensure sentiment score is a float between 0 and 1
        try:
            result["Sentiment Score"] = float(result["Sentiment Score"])
            result["Sentiment Score"] = max(0.0, min(1.0, result["Sentiment Score"]))
        except (ValueError, TypeError):
            result["Sentiment Score"] = 0.5
            
        # Ensure sentiment label is valid
        if result["Sentiment Label"] not in ["Positive", "Negative", "Neutral"]:
            result["Sentiment Label"] = "Neutral"
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        return {
            "Category": "Parse Error",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Reason": f"Failed to parse Claude response: {str(e)}",
            "Summary": response_text[:500] if 'response_text' in locals() else "Error",
        }
    except Exception as e:
        st.error(f"Claude API error: {e}")
        return {
            "Category": "API Error",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Reason": f"API call failed: {str(e)}",
            "Summary": "Unable to analyze text",
        }


# === Streamlit UI ===
st.title("Audio Transcription + Multi-Speaker Diarization")
uploaded_file = st.file_uploader("Upload a WAV/MP3/M4A file", type=["wav", "mp3", "m4a"])

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

    # Claude analysis
    full_text = result.get("text", "")
    if full_text:
        st.subheader("Analysis: Summary & Sentiment")
        with st.spinner("Analyzing with Claude..."):
            analysis = analyze_with_claude(full_text)

        st.write("**Category:**", analysis.get("Category", "N/A"))
        st.write("**Sentiment Score:**", analysis.get("Sentiment Score", "N/A"))
        st.write("**Sentiment Label:**", analysis.get("Sentiment Label", "N/A"))
        st.write("**Reason:**", analysis.get("Reason", "N/A"))
        st.subheader("Summary")
        st.text_area("Summary", analysis.get("Summary", ""), height=200)

