# import streamlit as st
# import tempfile
# import os
# import requests
# import json
# import re
# import unicodedata
# import time
# import httpx
# from pyannote.audio import Pipeline
# import whisper
# from anthropic import Anthropic

# # === Config ===
# ASSEMBLYAI_API_KEY = st.secrets["api_keys"]["assemblyai"]
# HUGGINGFACE_TOKEN = st.secrets["api_keys"]["huggingface"]
# CLAUDE_API_KEY = st.secrets["api_keys"]["anthropic"]

# client = Anthropic(api_key=CLAUDE_API_KEY)
# UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
# TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# # === Streamlit UI ===
# st.title("🎙️ Audio Transcription + Speaker Diarization")
# st.caption("Choose transcription engine and speaker detection method.")

# engine_choice = st.radio(
#     "Select engine:",
#     ["AssemblyAI (Transcription + Diarization)", "Whisper + Pyannote"]
# )

# uploaded_file = st.file_uploader("Upload a WAV/MP3/M4A file", type=["wav", "mp3", "m4a"])

# # === Helper functions ===
# def robust_sanitize_multilingual_text(raw_text: str) -> str:
#     if not raw_text:
#         return ""
#     text = raw_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
#     text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
#     text = text.replace("�", "")
#     text = unicodedata.normalize("NFC", text)
#     text = re.sub(r"[\u200B-\u200D\uFEFF]", " ", text)
#     text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def analyze_with_claude(text):
#     sanitized_text = robust_sanitize_multilingual_text(text)
#     prompt = f"""Analyze the sentiment and content of the following text and respond ONLY with valid JSON.

#     Text to analyze:
#     {sanitized_text}

#     Requirements:
#     1. Determine the overall sentiment (Positive, Negative, or Neutral)
#     2. Assign a sentiment score from 0.0 (most negative) to 1.0 (most positive), where 0.5 is neutral
#     3. Categorize the main topic/theme
#     4. Explain why you assigned this sentiment
#     5. Provide a brief summary

#     Respond with this exact JSON structure (no additional text):
#     {{
#     "Category": "the main topic or theme",
#     "Sentiment Score": 0.0,
#     "Sentiment Label": "Positive/Negative/Neutral",
#     "Reason": "explanation of sentiment determination",
#     "Summary": "brief summary of the text"
#     }}
#     Examples:
#         - "I love this product!" → Score: 0.9, Label: "Positive"
#         - "This is terrible and broken" → Score: 0.1, Label: "Negative"  
#         - "The meeting is scheduled for 3pm" → Score: 0.5, Label: "Neutral"
#     Return ONLY the JSON object, no markdown formatting or extra text."""

#     try:
#         response = client.messages.create(
#             model="claude-3-5-sonnet-20241022",
#             max_tokens=512,
#             temperature=0,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         response_text = response.content[0].text.strip()
#         response_text = re.sub(r"^```(json)?", "", response_text)
#         response_text = re.sub(r"```$", "", response_text).strip()
#         match = re.search(r"\{[\s\S]*\}", response_text)
#         json_str = match.group(0) if match else response_text
#         result = json.loads(json_str)
#         # Ensure required keys
#         for key in ["Category", "Sentiment Score", "Sentiment Label", "Reason", "Summary"]:
#             if key not in result:
#                 result[key] = "Unknown" if key != "Sentiment Score" else 0.5
#         result["Sentiment Score"] = float(result["Sentiment Score"])
#         result["Sentiment Score"] = max(0.0, min(1.0, result["Sentiment Score"]))
#         if result["Sentiment Label"] not in ["Positive", "Negative", "Neutral"]:
#             result["Sentiment Label"] = "Neutral"
#         return result
#     except Exception as e:
#         st.error(f"Claude API error: {e}")
#         return {
#             "Category": "API Error",
#             "Sentiment Score": 0.5,
#             "Sentiment Label": "Neutral",
#             "Reason": f"API call failed: {str(e)}",
#             "Summary": "Unable to analyze text",
#         }

# # === AssemblyAI helpers ===
# def upload_file_assembly(file_path):
#     headers = {"authorization": ASSEMBLYAI_API_KEY}
#     with open(file_path, "rb") as f:
#         with httpx.Client(http2=False, timeout=None) as client:
#             response = client.post(UPLOAD_URL, headers=headers, content=f)
#             response.raise_for_status()
#             return response.json()["upload_url"]

# def transcribe_assemblyai(audio_url, speaker_labels=False):
#     headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
#     data = {"audio_url": audio_url, "speaker_labels": speaker_labels, "language_detection": True}
#     r = requests.post(TRANSCRIPT_URL, headers=headers, json=data)
#     r.raise_for_status()
#     transcript_id = r.json()["id"]
#     polling_url = f"{TRANSCRIPT_URL}/{transcript_id}"
#     while True:
#         r = requests.get(polling_url, headers=headers)
#         res = r.json()
#         if res["status"] == "completed":
#             return res
#         elif res["status"] == "error":
#             st.error(res.get("error", "Unknown error"))
#             return {}
#         time.sleep(3)

# # === Whisper + Pyannote helpers ===
# @st.cache_resource
# def load_diarization_model():
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=HUGGINGFACE_TOKEN
#     )
#     return pipeline

# def diarize_with_pyannote(audio_path):
#     pipeline = load_diarization_model()
#     diarization = pipeline(audio_path)
#     segments = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         segments.append({"start": round(turn.start,2), "end": round(turn.end,2), "speaker": f"SPEAKER_{speaker}"})
#     return segments

# def normalize_speakers(segments):
#     speaker_map = {}
#     next_label = ord("A")
#     normalized = []
#     for seg in segments:
#         orig_speaker = seg["speaker"]
#         if orig_speaker not in speaker_map:
#             speaker_map[orig_speaker] = f"Speaker {chr(next_label)}"
#             next_label += 1
#         normalized.append({
#             "speaker": speaker_map[orig_speaker],
#             "text": seg["text"],
#             "start": seg.get("start"),
#             "end": seg.get("end")
#         })
#     return normalized

# # === File processing ===
# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(uploaded_file.read())
#         tmp.flush()
#         path = tmp.name

#     try:
#         if engine_choice.startswith("AssemblyAI"):
#             audio_url = upload_file_assembly(path)
#             transcript_result = transcribe_assemblyai(audio_url, speaker_labels=True)
#             full_text = transcript_result.get("text", "")
#             # Extract speaker segments from AssemblyAI
#             segments = []
#             words = transcript_result.get("words", [])
#             if words:
#                 current_speaker = None
#                 current_text = []
#                 for w in words:
#                     speaker = w.get("speaker")
#                     if speaker != current_speaker:
#                         if current_speaker:
#                             segments.append({
#                                 "speaker": current_speaker,
#                                 "text": " ".join(current_text)
#                             })
#                         current_speaker = speaker
#                         current_text = [w["text"]]
#                     else:
#                         current_text.append(w["text"])
#                 if current_text:
#                     segments.append({"speaker": current_speaker, "text": " ".join(current_text)})
#             speakers = normalize_speakers(segments)

#         else:  # Whisper + Pyannote
#             st.info("Transcribing with Whisper...")
#             whisper_model = whisper.load_model("large")
#             result = whisper_model.transcribe(path, task='transcribe', language="hi")
#             full_text = result["text"]
#             print("full trxt",full_text)
#             st.info("Diarizing with Pyannote...")
#             speaker_segments = diarize_with_pyannote(path)

#             # Align text with speakers (simple split by sentences)
#             sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_text) if s.strip()]
#             segments = []
#             for i, sent in enumerate(sentences):
#                 seg = speaker_segments[i % len(speaker_segments)]
#                 seg["text"] = sent
#                 segments.append(seg)
#             speakers = normalize_speakers(segments)

#         # Display transcription
#         st.subheader("📄 Full Transcription")
#         st.text_area("Complete Text", full_text, height=200)

#         # Display speaker conversation
#         st.subheader("👥 Speaker-wise Conversation")
#         for utt in speakers:
#             col1, col2 = st.columns([1, 5])
#             with col1:
#                 st.markdown(f"**{utt['speaker']}**")
#             with col2:
#                 st.write(utt['text'])
#             st.divider()

#         # Claude analysis
#         if full_text:
#             st.subheader("🤖 Analysis: Summary & Sentiment")
#             with st.spinner("Analyzing with Claude..."):
#                 analysis = analyze_with_claude(full_text)

#             st.markdown(f"""
#         **Category:** {analysis.get("Category", "N/A")}

#         **Sentiment Score:** {analysis.get("Sentiment Score", 0.5):.2f}  
#         **Sentiment Label:** {analysis.get("Sentiment Label", "Neutral")}

#         **Reason:** {analysis.get("Reason", "N/A")}

#         **Summary**  
#         {analysis.get("Summary", "No summary available")}
#         """)

#     except Exception as e:
#         st.error(f"Error processing audio: {e}")
#     finally:
#         if os.path.exists(path):
#             os.unlink(path)
from io import BytesIO
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
from elevenlabs.client import ElevenLabs


# === Config ===
ASSEMBLYAI_API_KEY = st.secrets["api_keys"]["assemblyai"]
UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
ELEVEN_API_KEY = st.secrets["api_keys"]["elevenlabs"]
elevenlabs = ElevenLabs(api_key=ELEVEN_API_KEY)

CLAUDE_API_KEY = st.secrets["api_keys"]["anthropic"]
client = Anthropic(api_key=CLAUDE_API_KEY)
# === Helper functions ===
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
        st.warning(f"enoising failed: {e}, using original audio.")
        return input_path

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
                st.success(f"✅ Detected: **{detected_lang.upper()}** (confidence: {confidence:.2f})")
                return detected_lang
            elif res["status"] == "error":
                st.warning("⚠️ Language detection failed, defaulting to English")
                return "en"
            time.sleep(2)

def format_speaker_diarization(transcription):
    """
    Format ElevenLabs transcription with speaker diarization.
    Groups consecutive words by the same speaker into dialogue turns.
    Fixes spacing and punctuation issues.
    """
    if not hasattr(transcription, 'words') or not transcription.words:
        return []

    formatted_lines = []
    current_speaker = None
    current_text = []

    for word_obj in transcription.words:
        # Skip spacing tokens if any
        if getattr(word_obj, "type", None) == "spacing":
            continue

        speaker = getattr(word_obj, "speaker_id", None)
        word = getattr(word_obj, "text", "")

        # Speaker changed — push previous segment
        if speaker != current_speaker:
            if current_speaker is not None and current_text:
                # Join words with spaces and fix spacing issues
                text = " ".join(current_text).strip()
                text = re.sub(r"\s+([,.!?।])", r"\1", text)  # no space before punctuation
                text = re.sub(r"([,.!?।])([^\s])", r"\1 \2", text)  # add space after punctuation
                speaker_label = current_speaker.replace("speaker_", "Speaker ")
                formatted_lines.append({
                    "speaker": speaker_label,
                    "text": text
                })
            current_speaker = speaker
            current_text = [word]
        else:
            current_text.append(word)

    # Add the last dialogue turn
    if current_speaker is not None and current_text:
        text = " ".join(current_text).strip()
        text = re.sub(r"\s+([,.!?।])", r"\1", text)
        text = re.sub(r"([,.!?।])([^\s])", r"\1 \2", text)
        speaker_label = current_speaker.replace("speaker_", "Speaker ")
        formatted_lines.append({
            "speaker": speaker_label,
            "text": text
        })

    return formatted_lines


def transcribe_elevenlabs(file_path, language_code):
    """Transcribe via ElevenLabs and format speaker diarization if available."""
    with open(file_path, "rb") as f:
        audio_data = BytesIO(f.read())

    transcription = elevenlabs.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1",
        language_code=language_code,
        diarize=True
    )

    # Format diarization output if available
    utterances = format_speaker_diarization(transcription)
    
    # Return unified result structure
    result = {
        "text": getattr(transcription, "text", ""),
        "utterances": utterances
    }
    return result


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

    # Collapse redundant whitespace
    return re.sub(r"\s+", " ", text).strip()


def extract_json_from_text(response_text: str) -> dict:
    """
    Robust JSON extraction and repair for AI responses.
    """
    response_text = response_text.strip()
    response_text = re.sub(r"^```(json)?", "", response_text)
    response_text = re.sub(r"```$", "", response_text).strip()

    # Try to extract the JSON block
    match = re.search(r"\{[\s\S]*\}", response_text)
    json_str = match.group(0) if match else response_text

    # Clean up known issues
    json_str = (
        json_str.replace("\\'", "'")
        .replace('\\"', '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt minor repairs: missing commas or quotes
        repaired = re.sub(r"(\w)\"(\w)", r'\1", "\2', json_str)
        repaired = repaired.replace("\\", "")
        try:
            import json5
            return json5.loads(repaired)
        except Exception:
            return {"Category": "Parse Error", "Sentiment Label": "Neutral", "Reason": "Invalid JSON returned by Claude", "Raw": json_str[:4000]}

def analyze_with_claude(text):
    sanitized_text = robust_sanitize_multilingual_text(text)
    prompt = f"""Analyze this conversation and return your analysis as a JSON object.

<conversation>
{sanitized_text}
</conversation>

Analyze the conversation for:
1. Overall sentiment (Positive/Negative/Neutral) and score (0.0 to 1.0)
2. Main topic/category
3. How the conversation flows between speakers
4. Emotional progression for each speaker with transitions
5. Speaker characteristics (mood, likely gender, approximate age)
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
  "Category": "main topic here",
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
    {{"Speaker": "Speaker A", "Mood": "happy", "Gender": "Male", "Age": "30-40"}}
  ],
  "Reason": "explanation for sentiment determination",
  "Summary": "brief summary of conversation"
}}

Make sure your output is strictly valid JSON and passes json.loads() in Python without any errors.
Output the JSON now:"""

    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
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

        result = json.loads(json_str)

        required_keys = [
            "Category",
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
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        return {
            "Category": "Parse Error",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Conversation Flow": "N/A",
            "Reason": f"Failed to parse Claude response: {str(e)}",
            "Summary": response_text[:500] if 'response_text' in locals() else "Error",
        }
    except Exception as e:
        st.error(f"Claude API error: {e}")
        return {
            "Category": "API Error",
            "Sentiment Score": 0.5,
            "Sentiment Label": "Neutral",
            "Conversation Flow": "N/A",
            "Reason": f"API call failed: {str(e)}",
            "Summary": "Unable to analyze text",
        }

# === Streamlit UI ===
st.title("🎙️ Audio Transcription + Multi-Speaker Diarization")
provider = st.selectbox("Select STT Provider", ["AssemblyAI", "ElevenLabs"])
st.caption("Automatic language detection → Accurate transcription with speaker identification")

uploaded_file = st.file_uploader("Upload a WAV/MP3/M4A file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        path = tmp.name
    
    try:
        # Step 1: Upload file
        denoised_path = denoise_audio(path)
        with st.spinner("⬆️ Uploading audio..."):
            audio_url = upload_file(path)
            st.success("✅ Upload complete!")
        
        # Step 2: Detect language
        detected_language = detect_language(audio_url)
        if provider == "AssemblyAI":
        
        # Step 3: Transcribe with detected language
            result = transcribe(audio_url, language_code=detected_language)
        else:
            result = transcribe_elevenlabs(denoised_path, detected_language)
        
        if result:
            # Display full transcription
            st.subheader("📄 Full Transcription")
            full_text = result.get("text", "")
            st.text_area("Complete Text", full_text, height=300)

            # Display speaker conversation
            st.subheader("👥 Speaker-wise Conversation")
            speakers = normalize_speakers(result.get("utterances", []))
            
            if speakers:
                for utt in speakers:
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.markdown(f"**{utt['speaker']}**")
                    with col2:
                        st.write(utt['text'])
                    st.divider()
            else:
                st.info("No speaker diarization data available")

            # Claude analysis
            if full_text:
                st.subheader("🤖 AI Analysis: Summary & Sentiment")
                with st.spinner("Analyzing with Claude..."):
                    analysis = analyze_with_claude(full_text)

        st.write("**Category:**", analysis.get("Category", "N/A"))
        st.write("**Sentiment Score:**", analysis.get("Sentiment Score", "N/A"))
        st.write("**Sentiment Label:**", analysis.get("Sentiment Label", "N/A"))
        st.write("**Conversation Flow:**", analysis.get("Conversation Flow", "N/A"))
        st.write("**Reason:**", analysis.get("Reason", "N/A"))
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

        st.subheader("Speaker Analysis")
        speaker_analysis = analysis.get("Speaker Analysis", [])
        if speaker_analysis:
            for sa in speaker_analysis:
                st.write(f"{sa['Speaker']} → Mood: {sa.get('Mood','N/A')}, Gender: {sa.get('Gender','N/A')}")
        else:
            st.info("No speaker analysis available")

        st.subheader("Summary")
        st.text_area("Summary", analysis.get("Summary", ""), height=200)
                
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
    finally:
        if os.path.exists(path):
            os.unlink(path)

