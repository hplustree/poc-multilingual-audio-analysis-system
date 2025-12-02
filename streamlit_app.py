from functools import wraps
import zipfile
import streamlit as st
import requests
import tempfile
import os
from anthropic import Anthropic
import time
import json
import re
import unicodedata
import httpx
import noisereduce as nr
import soundfile as sf
import numpy as np
import concurrent.futures
import pandas as pd
import threading
import queue
import whisper
import librosa
# === Config ===
ASSEMBLYAI_API_KEY = st.secrets["api_keys"]["assemblyai"]
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
CLAUDE_API_KEY = st.secrets["api_keys"]["anthropic"]
client = Anthropic(api_key=CLAUDE_API_KEY)
SONIOX_BASE_URL = "https://api.soniox.com/v1"

SONIOX_API_KEY= st.secrets["api_keys"]["soniox"]

# Language code to name mapping
LANGUAGE_NAMES = {
    "en": "ENGLISH",
    "hi": "HINDI",
    "ml": "MALAYALAM",
}

def format_language_display(lang_code, confidence=None):
    """Format language as 'CODE-NAME' with optional confidence score."""
    lang_code_upper = lang_code.upper()
    lang_name = LANGUAGE_NAMES.get(lang_code.lower(), lang_code.upper())
    display = f"{lang_code_upper}-{lang_name}"
    if confidence is not None:
        display += f" (confidence: {confidence:.2f})"
    return display

# === Helper functions ===
if "zip_state" not in st.session_state:
    st.session_state["zip_state"] = {
        "status": "idle",
        "results": [],
        "total": 0,
        "futures": None
    }

# === Claude Queue (Safe Rate-Limited Worker) ===
CLAUDE_MIN_DELAY = 13
claude_queue = queue.Queue()

def claude_worker():
    """
    Dedicated worker that processes Claude API requests sequentially.
    Ensures we never exceed Anthropic rate limits.
    """
    last_call_time = 0

    while True:
        item = claude_queue.get()

        if item is None:
            break  # stop thread

        text, detected_language, callback = item

        # Enforce safe spacing between API calls
        elapsed = time.time() - last_call_time
        if elapsed < CLAUDE_MIN_DELAY:
            time.sleep(CLAUDE_MIN_DELAY - elapsed)

        result = analyze_with_claude(text, detected_language)
        last_call_time = time.time()

        # Return result to waiting thread
        callback(result)
        claude_queue.task_done()

# Start worker thread
threading.Thread(target=claude_worker, daemon=True).start()

def retry_api(max_retries=5, base_delay=1, backoff=2, extra_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                # Slow down EVERY request to avoid rate-limit
                if extra_delay > 0:
                    time.sleep(extra_delay)

                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    err_msg = str(e)
                    retryable = (
                        "429" in err_msg or
                        "rate limit" in err_msg.lower() or
                        "503" in err_msg or
                        "500" in err_msg or
                        isinstance(e, httpx.HTTPError) or
                        isinstance(e, requests.exceptions.RequestException)
                    )

                    if not retryable:
                        raise

                    attempt += 1
                    wait = base_delay * (backoff ** (attempt - 1))
                    st.warning(f"⏳ Retry {attempt}/{max_retries}: {err_msg}. Waiting {wait:.1f}s...")
                    time.sleep(wait)

            raise RuntimeError(f"API failed after {max_retries} retries")
        return wrapper
    return decorator

@retry_api(extra_delay=1)
def upload_file(file_path):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(file_path, "rb") as f:
        with httpx.Client(http2=False, timeout=None) as http_client:
            response = http_client.post(
                UPLOAD_URL,
                headers=headers,
                content=f
            )
            response.raise_for_status()
            return response.json()["upload_url"]

def denoise_audio(input_path):
    """Remove background noise using spectral gating (noisereduce)."""
    st.info("Denoising audio...")
    try:
        data, samplerate = sf.read(input_path)
        if len(data.shape) > 1:  # stereo → mono
            data = np.mean(data, axis=1)
        reduced_noise = nr.reduce_noise(y=data, sr=samplerate)
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(out_path, reduced_noise, samplerate)
        st.success("Noise reduced successfully!")
        return out_path
    except Exception as e:
        st.warning(f"denoising failed: {e}, using original audio.")
        return input_path

@retry_api(extra_delay=1)
def detect_language(audio_url):
    """First pass: detect the dominant language"""
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    data = {
        "audio_url": audio_url,
        "language_detection": True
    }
    r = requests.post(TRANSCRIPT_URL, headers=headers, json=data)
    r.raise_for_status()
    transcript_id = r.json()["id"]
    polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
    with st.spinner("🔍 Detecting language..."):
        while True:
            r = requests.get(polling_url, headers=headers)
            res = r.json()
            if res["status"] == "completed":
                detected_lang = res.get("language_code", "en")
                confidence = res.get("language_confidence", 0)
                lang_display = format_language_display(detected_lang, confidence)
                st.success(f"✅ Detected: **{lang_display}**")
                return detected_lang, confidence
            elif res["status"] == "error":
                st.warning("⚠️ Language detection failed, defaulting to English")
                return "en", 0.0
            time.sleep(2)

@retry_api(extra_delay=1)
def transcribe(audio_url, language_code):
    """Second pass: transcribe with detected language"""
    headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
    data = {
        "audio_url": audio_url, 
        "speaker_labels": True, 
        "language_code": language_code
    }
    r = requests.post(TRANSCRIPT_URL, headers=headers, json=data)
    r.raise_for_status()
    transcript_id = r.json()["id"]

    polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
    with st.spinner(f"📝 Transcribing in **{language_code.upper()}**..."):
        while True:
            r = requests.get(polling_url, headers=headers)
            res = r.json()
            if res["status"] == "completed":
                return res
            elif res["status"] == "error":
                st.error(res.get("error", "Unknown error"))
                return {}
            time.sleep(3)

@retry_api(extra_delay=1)
def soniox_upload(filepath):
    with httpx.Client(http2=True, timeout=None) as client:
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f, "application/octet-stream")}
            r = client.post(
                f"{SONIOX_BASE_URL}/files",
                headers={"Authorization": f"Bearer {SONIOX_API_KEY}"},
                files=files
            )
    r.raise_for_status()
    return r.json()["id"]

@retry_api(extra_delay=1)
def soniox_transcribe(file_id, lang_hints = None):


    payload = {
        "file_id": file_id,
        "model": "stt-async-v3",
        "language_hints": ["en","hi","ml"],
        "enable_language_identification": True,
        "enable_speaker_diarization": True,
        "speaker_diarization_max_speakers": 2,
        "enable_word_timestamps": True,
        "enable_punctuation": True
    }
    if lang_hints:
        payload["language_hints"] = [lang_hints]

    with httpx.Client(http2=True, timeout=None) as client:
        r = client.post(
            f"{SONIOX_BASE_URL}/transcriptions",
            headers={
                "Authorization": f"Bearer {SONIOX_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )

    r.raise_for_status()
    return r.json()["id"]


def soniox_poll(job_id):
    with httpx.Client(http2=True, timeout=None) as client:
        while True:
            r = client.get(
                f"{SONIOX_BASE_URL}/transcriptions/{job_id}",
                headers={"Authorization": f"Bearer {SONIOX_API_KEY}"}
            )
            data = r.json()
            print("poll:", data)

            if data["status"] == "completed":
                return
            if data["status"] == "failed":
                raise RuntimeError(f"Soniox failed: {data}")

            time.sleep(2)

def soniox_get_transcript(job_id):
    with httpx.Client(http2=True, timeout=None) as client:
        r = client.get(
            f"{SONIOX_BASE_URL}/transcriptions/{job_id}/transcript",
            headers={"Authorization": f"Bearer {SONIOX_API_KEY}"}
        )
    r.raise_for_status()
    return r.json()

def soniox_delete_file(file_id):
    with httpx.Client(http2=True, timeout=None) as client:
        r = client.delete(
            f"{SONIOX_BASE_URL}/files/{file_id}",
            headers={"Authorization": f"Bearer {SONIOX_API_KEY}"}
        )
        if r.status_code != 204:
            print("Warning: could not delete file:", r.text)

def soniox_delete_transcription(job_id):
    with httpx.Client(http2=True, timeout=None) as client:
        r = client.delete(
            f"{SONIOX_BASE_URL}/transcriptions/{job_id}",
            headers={"Authorization": f"Bearer {SONIOX_API_KEY}"}
        )
        if r.status_code != 204:
            print("Warning: could not delete transcription:", r.text)


def transcribe_soniox(file_path, lang_hints = None):
    file_id = soniox_upload(file_path)
    job_id = soniox_transcribe(file_id, lang_hints=lang_hints)
    soniox_poll(job_id)

    final = soniox_get_transcript(job_id)
    try:
        soniox_delete_file(file_id=file_id)
        soniox_delete_transcription(job_id=job_id)
    except Exception as e:
        raise e
    tokens = final.get("tokens", [])
    utterances = []
    current_speaker = None
    current_text = ""
    start_time = None

    def clean_text(txt):
        # Fix extra spaces and punctuation spacing
        txt = re.sub(r"\s+", " ", txt)
        txt = re.sub(r"\s+([,.।?!])", r"\1", txt)
        return txt.strip()

    for t in tokens:
        sp = t.get("speaker")
        grapheme = t.get("text", "")

        # Skip if token is blank
        if grapheme is None or grapheme.strip() == "":
            continue

        # Speaker switch → close segment
        if sp != current_speaker:
            if current_speaker is not None and current_text.strip():
                utterances.append({
                    "speaker": f"Speaker {current_speaker}",
                    "start": start_time / 1000 if start_time else None,
                    "end": t["start_ms"] / 1000,
                    "text": clean_text(current_text)
                })
            current_speaker = sp
            current_text = grapheme
            start_time = t["start_ms"]
        else:
            if re.match(r"[,.।?!]", grapheme):
                current_text += grapheme
            else:
                current_text += grapheme

    # Close last segment
    if current_speaker and current_text.strip():
        utterances.append({
            "speaker": f"Speaker {current_speaker}",
            "start": start_time / 1000 if start_time else None,
            "end": tokens[-1]["end_ms"] / 1000 if tokens else None,
            "text": clean_text(current_text)
        })

    return {
        "text": final.get("text", "").strip(),
        "utterances": utterances
    }
@st.cache_resource
def load_whisper_small_model():
    return whisper.load_model("small")

def detect_language_whisper(file_path):
    st.info("Detecting language (Whisper Small)...")

    try:
        whisper_model = load_whisper_small_model()

        data, samplerate = sf.read(file_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 🔥 Convert dtype before resample
        data = data.astype(np.float32)

        # Resample if needed
        if samplerate != 16000:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=16000).astype(np.float32)

        # Whisper expects float32 16000hz mono
        audio = whisper.pad_or_trim(data)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

        _, probs = whisper_model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        confidence = float(probs[detected_lang])

        st.success(f"Detected Language: **{detected_lang.upper()}** ({confidence:.2f})")
        return detected_lang, confidence

    except Exception as e:
        st.warning(f"⚠️ Whisper Small detection failed: {e}. Defaulting to English.")
        return "en", 0.5

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
    Cleans and normalizes multilingual text (Tamil, Malayalam, Hindi, English).
    Removes invalid Unicode characters and invisible artifacts.
    """
    if not raw_text:
        return ""
    
    # Replace or remove problematic encodings
    text = (
        raw_text.encode("utf-8", errors="replace")
        .decode("utf-8", errors="replace")
        .replace("�", "")
    )

    # Remove control & zero-width characters
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", " ", text)
    
    # Normalize Unicode composition
    text = unicodedata.normalize("NFC", text)

    # Replace formatting controls
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    # Collapse redundant whitespace
    return re.sub(r"\s+", " ", text).strip()

def clean_json_unicode(text: str) -> str:
    import re, unicodedata
    # Normalize Unicode composition
    text = unicodedata.normalize("NFC", text)
    # Remove control and zero-width chars
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    # Replace fancy quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return text
def extract_json_from_text(response_text: str) -> dict:
    import json5
    text = clean_json_unicode(response_text.strip())

    # Extract JSON block
    match = re.search(r"\{[\s\S]*", text)
    if not match:
        return {}

    json_str = match.group(0)

    # 1. Try parsing raw JSON with json5
    try:
        return json5.loads(json_str)
    except Exception:
        pass

    # -----------------------------
    # 2. JSON REPAIR STAGE (very important)
    # -----------------------------
    repaired = json_str

    # Remove trailing commas (VERY common cause of json5 failure too)
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)

    # Fix unbalanced braces { }
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    if close_braces < open_braces:
        repaired += "}" * (open_braces - close_braces)

    # Fix unbalanced brackets [ ]
    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if close_brackets < open_brackets:
        repaired += "]" * (open_brackets - close_brackets)

    # Try json5 again after repair
    try:
        return json5.loads(repaired)
    except Exception:
        pass
    partial = {}

    fields = [
        "Main Category",
        "Sub Category",
        "Language",
        "Sentiment Score",
        "Sentiment Label",
        "Conversation Flow",
        "Reason",
        "Summary",
    ]

    for f in fields:
        m = re.search(rf'"{f}"\s*:\s*"([^"]*)"', json_str)
        if m:
            partial[f] = m.group(1)

    # Fill defaults for missing entries
    for f in fields:
        partial.setdefault(f, "")

    return partial


@retry_api(extra_delay=2)
def analyze_with_claude(text, detected_language="en"):
    sanitized_text = robust_sanitize_multilingual_text(text)
    lang_display = format_language_display(detected_language)
    prompt = f"""Analyze this conversation and return your analysis as a JSON object.

<conversation>
{sanitized_text}
</conversation>

The conversation language is: {lang_display}

Analyze the conversation for:
1. Overall sentiment (Positive/Negative/Neutral) and score (0.0 to 1.0)
2. Main category and sub-category (e.g., Main: "Customer Service", Sub: "Billing Inquiry")
3. How the conversation flows between speakers
4. Emotional progression for each speaker with transitions
5. Speaker characteristics (mood, likely gender)
6. Brief summary

IMPORTANT FORMATTING RULES:
- Output ONLY a valid JSON object
- Do NOT use markdown code blocks or backticks
- Do NOT add any text before or after the JSON
- Ensure all strings are properly escaped
- Use double quotes for all keys and string values
- Keep all text values on single lines (no line breaks within strings)
- Ensure all brackets and braces are properly closed

Return this exact JSON structure:
{{
  "Main Category": "broad category here",
  "Sub Category": "specific sub-category here",
  "Language": "{lang_display}",
  "Sentiment Score": 0.75,
  "Sentiment Label": "Positive",
  "Conversation Flow": "single line description of how conversation developed",
  "Emotion Flow": [
    {{
      "Speaker": "Speaker A",
      "Emotions": ["Emotion1", "Emotion2"],
      "Transitions": [
        {{
          "From": "Emotion1",
          "To": "Emotion2",
          "How": "description of how tone changed",
          "Why": "reason for change",
          "Where": "context or quote"
        }}
      ]
    }}
  ],
  "Speaker Analysis": [
    {{"Speaker": "Speaker A", "Mood": "happy", "Gender": "Male"}}
  ],
  "Reason": "explanation for sentiment determination",
  "Summary": "brief summary of conversation"
}}

Make sure your output is strictly valid JSON and passes json.loads() in Python without any errors.
Output the JSON now:"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=8000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text.strip()
        response_text = re.sub(r"^```(json)?", "", response_text)
        response_text = re.sub(r"```$", "", response_text).strip()

        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            json_str = match.group(0)
        else:
            json_str = response_text

        result = extract_json_from_text(response_text)

        required_keys = [
            "Main Category",
            "Sub Category",
            "Language",
            "Sentiment Score",
            "Sentiment Label",
            "Conversation Flow",
            "Reason",
            "Summary"
        ]
        for key in required_keys:
            if key not in result:
                result[key] = "Unknown" if key != "Sentiment Score" else 0.5

        try:
            result["Sentiment Score"] = float(result["Sentiment Score"])
            result["Sentiment Score"] = max(0.0, min(1.0, result["Sentiment Score"]))
        except (ValueError, TypeError):
            result["Sentiment Score"] = 0.5

        if result["Sentiment Label"] not in ["Positive", "Negative", "Neutral"]:
            result["Sentiment Label"] = "Neutral"
        result["Language"] = format_language_display(detected_language)
        return result
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        return {
            "Main Category": "Parse Error",
            "Sub Category": "N/A",
            "Language": "Unknown",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Conversation Flow": "N/A",
            "Reason": f"Failed to parse Claude response: {str(e)}",
            "Summary": response_text[:500] if 'response_text' in locals() else "Error",
        }
    except Exception as e:
        st.error(f"Claude API error: {e}")
        return {
            "Main Category": "API Error",
            "Sub Category": "N/A",
            "Language": "Unknown",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Conversation Flow": "N/A",
            "Reason": f"API call failed: {str(e)}",
            "Summary": "Unable to analyze text",
        }

# === Streamlit UI ===
st.title("🎙️ Batch Audio Transcription + Multi-Speaker Emotion & Sentiment Analysis")
provider = st.selectbox("Select STT Provider", ["AssemblyAI", "Soniox"])
st.caption(
    "Upload either a single audio file or a ZIP folder of multiple audio files. "
    "The system will detect the language, transcribe, and perform emotion, sentiment, and speaker analysis."
)

uploaded_file = st.file_uploader("Upload audio file or ZIP folder", type=["zip", "wav", "mp3", "m4a"], key="uploader")

def process_audio_file(file_path, provider):
    """Single-file pipeline using all your existing helper functions."""
    try:
        denoised_path = denoise_audio(file_path)
        audio_url = upload_file(denoised_path)

        # Transcribe
        if provider == "Soniox":
            with st.spinner("🎧 Transcribing with Soniox..."):
                detected_language, lang_conf = detect_language_whisper(denoised_path)
                result = transcribe_soniox(denoised_path, detected_language)
        else:
            detected_language, lang_conf = detect_language(audio_url)
            with st.spinner(f"📝 Transcribing with AssemblyAI in {detected_language.upper()}..."):
                result = transcribe(audio_url, language_code=detected_language)

        # Analyze
        full_text = result.get("text", "")
        analysis_container = {"value": None}
        event = threading.Event()

        def on_done(result):
            analysis_container["value"] = result
            event.set()

        if full_text:
            claude_queue.put((full_text, detected_language, on_done))
            event.wait()  # wait for Claude result

        analysis = analysis_container["value"] if full_text else {}
        if analysis and lang_conf is not None:
            analysis["Language Confidence"] = lang_conf

        # Include full diarization but without timestamps
        utterances = result.get("utterances", [])
        for utt in utterances:
            utt.pop("start", None)
            utt.pop("end", None)

        # Replace numeric speakers with readable labels directly
        utterances = normalize_speakers(utterances)

        return {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "transcript": full_text,
            "utterances": utterances,
            "analysis": analysis,
            "meta": {
                "provider": provider,
                "total_segments": len(utterances),
                "detected_language": detected_language,
                "detected_language_confidence": lang_conf
            },
            "status": "success",
        }
    except Exception as e:
        return {"filename": os.path.basename(file_path), "status": "error", "error": str(e)}


# === Cache & Process Folder ===
if uploaded_file and uploaded_file.name.lower().endswith((".wav", ".mp3", ".m4a")):
    st.session_state.pop("results", None)
    st.session_state.pop("last_processed_key", None)
    st.info(f"🎧 Processing single file: **{uploaded_file.name}** ...")

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        temp_audio_path = tmp.name

    result = process_audio_file(temp_audio_path, provider)

    if result["status"] == "success":
        analysis = result["analysis"]
        st.success("✅ File processed successfully!")
        lang_code = result['meta']['detected_language']
        lang_conf = result['meta'].get('detected_language_confidence')
        lang_display = format_language_display(lang_code, lang_conf)
        st.write(f"**Detected Language:** {lang_display}")

        st.subheader("📄 Full Transcription")
        single_key = f"single_{uploaded_file.name}_{uploaded_file.size}"
        st.text_area(
            "Complete Text",
            result["transcript"],
            height=200,
            key=f"complete_text_{single_key}"
        )

        st.subheader("👥 Speaker Diarization")
        if result["utterances"]:
            for utt in result["utterances"]:
                st.markdown(f"**{utt['speaker']}**: {utt['text']}")
                st.divider()
        else:
            st.info("No speaker diarization available.")

        st.subheader("🤖 AI Analysis: Sentiment & Summary")
        st.write(f"**Language:** {analysis.get('Language', 'N/A')}")
        st.write(f"**Main Category:** {analysis.get('Main Category', 'N/A')}")
        st.write(f"**Sub Category:** {analysis.get('Sub Category', 'N/A')}")
        st.write(f"**Sentiment Score:** {analysis.get('Sentiment Score', 'N/A')}")
        st.write(f"**Sentiment Label:** {analysis.get('Sentiment Label', 'N/A')}")
        st.write(f"**Conversation Flow:** {analysis.get('Conversation Flow', 'N/A')}")
        st.write(f"**Reason:** {analysis.get('Reason', 'N/A')}")

        emotion_flow = analysis.get("Emotion Flow", [])
        if isinstance(emotion_flow, list) and emotion_flow:
            st.subheader("🧠 Emotion Flow Analysis")
            for speaker_data in emotion_flow:
                st.markdown(f"**{speaker_data['Speaker']}**: {' → '.join(speaker_data.get('Emotions', []))}")
                for t in speaker_data.get("Transitions", []):
                    st.write(f"• **{t['From']} → {t['To']}**")
                    st.caption(f"_How:_ {t['How']} | _Why:_ {t['Why']} | _Where:_ {t['Where']}")
        else:
            st.info("No detailed emotion flow available.")

        st.subheader("🗣️ Speaker Analysis")
        speaker_analysis = analysis.get("Speaker Analysis", [])
        if speaker_analysis:
            for sa in speaker_analysis:
                st.write(f"{sa['Speaker']} → Mood: {sa.get('Mood','N/A')}, Gender: {sa.get('Gender','N/A')}")
        else:
            st.info("No speaker analysis available.")

        st.subheader("🧾 Summary")
        st.text_area(
            "Summary",
            analysis.get("Summary", ""),
            height=150,
            key=f"summary_{single_key}"
        )

        json_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full JSON Result",
            data=json_bytes,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_analysis.json",
            mime="application/json",
            key=f"download_single_{single_key}"
        )
    else:
        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

# === Process ZIP of Multiple Files ===
elif uploaded_file and uploaded_file.name.lower().endswith(".zip"):
    st.info("📦 ZIP file detected — processing multiple audio files in parallel...")
    cache_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if "last_processed_key" not in st.session_state or st.session_state["last_processed_key"] != cache_key:
        st.session_state["last_processed_key"] = cache_key
        st.session_state["results"] = []

        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find all audio files
            audio_files = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith((".wav", ".mp3", ".m4a")):
                        abs_path = os.path.join(root, f)
                        rel_path = os.path.relpath(abs_path, extract_dir)
                        audio_files.append({"abs": abs_path, "rel": rel_path})

            if not audio_files:
                st.warning("No audio files found in ZIP.")
            else:
                st.success(f"📂 Found {len(audio_files)} files. Starting parallel processing...")
                progress = st.progress(0)
                results = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(audio_files))) as executor:
                    futures = {}
                    for item in audio_files:
                        time.sleep(1)
                        fut = executor.submit(process_audio_file, item["abs"], provider)
                        futures[fut] = item

                    total = len(futures)
                    completed = 0

                    for future in concurrent.futures.as_completed(futures):
                        completed += 1
                        progress.progress(completed / total)
                        item = futures[future]
                        rel = item["rel"]
                        abs_path = item["abs"]
                        file_name = os.path.basename(abs_path)
                        safe = rel.replace(os.sep, "__")
                        try:
                            result = future.result()
                            result["rel_path"] = rel

                            results.append(result)
                            st.session_state["results"] = results

                            if result["status"] == "success":
                                analysis = result["analysis"]

                                with st.expander(f"✅ {rel}", expanded=False):
                                    st.write(f"**Provider:** {result['meta']['provider']}")
                                    st.write(f"**Segments:** {result['meta']['total_segments']}")
                                    # Display detected language with formatted name
                                    lang_code = result['meta']['detected_language']
                                    lang_conf = result['meta'].get('detected_language_confidence')
                                    lang_display = format_language_display(lang_code, lang_conf)
                                    st.write(f"**Detected Language:** {lang_display}")
                                    st.subheader("📄 Full Transcription")
                                    st.text_area(
                                        "Complete Text",
                                        result["transcript"],
                                        height=200,
                                        key=f"complete_text_{safe}",
                                    )
                                    st.subheader("👥 Speaker Diarization")
                                    if result["utterances"]:
                                        for utt in result["utterances"]:
                                            st.markdown(f"**{utt['speaker']}**: {utt['text']}")
                                            st.divider()
                                    else:
                                        st.info("No speaker diarization available.")

                                    st.subheader("🤖 AI Analysis: Sentiment & Summary")
                                    st.write(f"**Language:** {analysis.get('Language', 'N/A')}")
                                    st.write(f"**Main Category:** {analysis.get('Main Category', 'N/A')}")
                                    st.write(f"**Sub Category:** {analysis.get('Sub Category', 'N/A')}")
                                    st.write(f"**Sentiment Score:** {analysis.get('Sentiment Score', 'N/A')}")
                                    st.write(f"**Sentiment Label:** {analysis.get('Sentiment Label', 'N/A')}")
                                    st.write(f"**Conversation Flow:** {analysis.get('Conversation Flow', 'N/A')}")
                                    st.write(f"**Reason:** {analysis.get('Reason', 'N/A')}")

                                    emotion_flow = analysis.get("Emotion Flow", [])
                                    if isinstance(emotion_flow, list) and emotion_flow:
                                        st.subheader("🧠 Emotion Flow Analysis")
                                        for speaker_data in emotion_flow:
                                            st.markdown(
                                                f"**{speaker_data['Speaker']}**: "
                                                f"{' → '.join(speaker_data.get('Emotions', []))}"
                                            )
                                            for t in speaker_data.get("Transitions", []):
                                                st.write(f"• **{t['From']} → {t['To']}**")
                                                st.caption(
                                                    f"_How:_ {t['How']} | "
                                                    f"_Why:_ {t['Why']} | "
                                                    f"_Where:_ {t['Where']}"
                                                )
                                    else:
                                        st.info("No detailed emotion flow available.")

                                    st.subheader("🗣️ Speaker Analysis")
                                    speaker_analysis = analysis.get("Speaker Analysis", [])
                                    if speaker_analysis:
                                        for sa in speaker_analysis:
                                            st.write(
                                                f"{sa['Speaker']} → "
                                                f"Mood: {sa.get('Mood', 'N/A')}, "
                                                f"Gender: {sa.get('Gender', 'N/A')}"
                                            )
                                    else:
                                        st.info("No speaker analysis available.")

                                    st.subheader("🧾 Summary")
                                    st.text_area(
                                        "Summary",
                                        analysis.get("Summary", ""),
                                        height=150,
                                        key=f"summary_{safe}"
                                    )

                                    json_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
                                    st.download_button(
                                        label="⬇️ Download Full JSON Result",
                                        data=json_bytes,
                                        file_name=f"{os.path.splitext(file_name)[0]}_analysis.json",
                                        mime="application/json",
                                        key=f"download_zip_{safe}"
                                    )

                            else:
                                st.error(f"❌ {rel} failed: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"❌ {rel} crashed: {str(e)}")

        success = [r for r in results if r["status"] == "success"]
        if success:
            df = pd.DataFrame([
                {
                    "File": r["filename"],
                    "Language": r["analysis"].get("Language", ""),
                    "Main Category": r["analysis"].get("Main Category", ""),
                    "Sub Category": r["analysis"].get("Sub Category", ""),
                    "Sentiment": r["analysis"].get("Sentiment Label", ""),
                    "Score": r["analysis"].get("Sentiment Score", ""),
                    "Summary": r["analysis"].get("Summary", "")
                }
                for r in success
            ])
            st.subheader("📊 Summary of All Files")
            st.dataframe(df)

else:
    st.info("👆 Please upload either a single audio file or a ZIP folder to begin processing.")
