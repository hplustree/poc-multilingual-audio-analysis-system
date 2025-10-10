# import os
# import time
# import tempfile
# import subprocess
# import requests
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from pydantic import BaseModel

# # FastAPI initialization
# app = FastAPI(title="Audio Transcription + Speaker Diarization API")

# # Set your AssemblyAI API key
# ASSEMBLYAI_API_KEY = '4600824a76e84ba5948711363fb84158'
# ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
# ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# # Response model
# class AnalysisResult(BaseModel):
#     language: str
#     transcription: str
#     speakers: list  # [{speaker, text, start, end}]

# # Optional: normalize audio
# def convert_audio_to_standard(input_path: str, output_path: str):
#     """Convert to mono, 16 kHz, 16-bit PCM WAV"""
#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", input_path,
#         "-ac", "1",
#         "-ar", "16000",
#         "-f", "wav",        # force WAV container
#         "-c:a", "pcm_s16le",
#         output_path
#     ]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# def upload_file_to_assemblyai(file_path: str) -> str:
#     """Upload audio file to AssemblyAI and get URL"""
#     print("📤 Uploading file to AssemblyAI...")
    
#     headers = {"authorization": ASSEMBLYAI_API_KEY}
    
#     with open(file_path, "rb") as f:
#         response = requests.post(
#             ASSEMBLYAI_UPLOAD_URL,
#             headers=headers,
#             data=f
#         )
    
#     if response.status_code != 200:
#         raise ValueError(f"Upload failed: {response.text}")
    
#     upload_url = response.json()["upload_url"]
#     print(f"✅ Upload successful: {upload_url}")
#     return upload_url

# def transcribe_with_diarization(audio_url: str):
#     """Submit transcription job with speaker diarization"""
#     print("🎧 Starting transcription with speaker diarization...")
    
#     headers = {
#         "authorization": ASSEMBLYAI_API_KEY,
#         "content-type": "application/json"
#     }
    
#     # Submit transcription request
#     json_data = {
#         "audio_url": audio_url,
#         "speaker_labels": True,
#         "language_detection": True
#     }
    
#     response = requests.post(
#         ASSEMBLYAI_TRANSCRIPT_URL,
#         headers=headers,
#         json=json_data
#     )
    
#     if response.status_code != 200:
#         raise ValueError(f"Transcription request failed: {response.text}")
    
#     transcript_id = response.json()["id"]
#     print(f"🆔 Job ID: {transcript_id}")
#     print("⏳ Waiting for transcription to complete...")
    
#     # Poll for completion
#     polling_url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
    
#     while True:
#         response = requests.get(polling_url, headers=headers)
#         result = response.json()
        
#         status = result["status"]
        
#         if status == "completed":
#             print("✅ Transcription completed!")
#             break
#         elif status == "error":
#             raise ValueError(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
        
#         time.sleep(3)
    
#     # Build speaker segments
#     speakers = []
#     for utt in result.get("utterances", []):
#         speakers.append({
#             "speaker": utt["speaker"],
#             "text": utt["text"],
#             "start": utt["start"],
#             "end": utt["end"]
#         })
    
#     # Combine into full transcript
#     full_text = result.get("text", "")
#     detected_language = result.get("language_code", "en")
    
#     return {
#         "language": detected_language,
#         "transcription": full_text,
#         "speakers": speakers
#     }

# @app.post("/analyze", response_model=AnalysisResult)
# async def analyze_audio(file: UploadFile = File(...)):
#     audio_path = None
#     converted_path = None
    
#     try:
#         # Save uploaded file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(await file.read())
#             tmp.flush()
#             audio_path = tmp.name

#         # Convert for consistency
#         converted_path = audio_path.replace(".wav", "_converted.wav")
#         convert_audio_to_standard(audio_path, converted_path)

#         # Upload to AssemblyAI
#         upload_url = upload_file_to_assemblyai(converted_path)

#         # Run transcription
#         result = transcribe_with_diarization(upload_url)
        
#         return result

#     except subprocess.CalledProcessError:
#         raise HTTPException(status_code=400, detail="Invalid or corrupted audio file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         # Cleanup temp files
#         if audio_path and os.path.exists(audio_path):
#             os.unlink(audio_path)
#         if converted_path and os.path.exists(converted_path):
#             os.unlink(converted_path)

import os
import time
import tempfile
import subprocess
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Audio Transcription + Multi-Speaker Diarization API")

ASSEMBLYAI_API_KEY = '4600824a76e84ba5948711363fb84158'
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

class SpeakerSegment(BaseModel):
    speaker: str
    text: str
    start: int
    end: int

class AnalysisResult(BaseModel):
    language: str
    transcription: str
    speakers: List[SpeakerSegment]

def convert_audio_to_standard(input_path: str, output_path: str):
    """Convert to mono, 16 kHz, 16-bit PCM WAV"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def split_audio(input_path: str, chunk_length_sec: int = 60) -> List[str]:
    """Split audio into chunks (in seconds)"""
    chunks = []
    total_duration_cmd = [
        "ffprobe", "-i", input_path, "-show_entries",
        "format=duration", "-v", "quiet", "-of", "csv=p=0"
    ]
    result = subprocess.run(total_duration_cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    start = 0

    while start < duration:
        chunk_path = f"{input_path}_chunk_{int(start)}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ss", str(start),
            "-t", str(chunk_length_sec),
            "-c", "copy",
            chunk_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        chunks.append(chunk_path)
        start += chunk_length_sec

    return chunks

def upload_file_to_assemblyai(file_path: str) -> str:
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    with open(file_path, "rb") as f:
        response = requests.post(
            ASSEMBLYAI_UPLOAD_URL,
            headers=headers,
            data=f
        )
    if response.status_code != 200:
        raise ValueError(f"Upload failed: {response.text}")
    return response.json()["upload_url"]

def transcribe_chunk(audio_url: str) -> dict:
    """Transcribe a single chunk with speaker diarization"""
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    json_data = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "language_detection": True
    }
    response = requests.post(ASSEMBLYAI_TRANSCRIPT_URL, headers=headers, json=json_data)
    if response.status_code != 200:
        raise ValueError(f"Transcription request failed: {response.text}")
    transcript_id = response.json()["id"]

    # Poll until done
    polling_url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
    while True:
        r = requests.get(polling_url, headers=headers)
        result = r.json()
        status = result["status"]
        if status == "completed":
            break
        elif status == "error":
            raise ValueError(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
        time.sleep(3)

    return result

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    audio_path = None
    converted_path = None
    chunk_paths = []
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp.flush()
            audio_path = tmp.name

        # Convert to standard WAV
        converted_path = audio_path.replace(".wav", "_converted.wav")
        convert_audio_to_standard(audio_path, converted_path)

        # Split into chunks
        chunk_paths = split_audio(converted_path, chunk_length_sec=60)

        all_speakers = []
        full_text = ""
        language_code = "en"

        # Transcribe each chunk
        for chunk in chunk_paths:
            upload_url = upload_file_to_assemblyai(chunk)
            result = transcribe_chunk(upload_url)

            # Append speaker segments
            for utt in result.get("utterances", []):
                all_speakers.append({
                    "speaker": utt["speaker"],
                    "text": utt["text"],
                    "start": utt["start"],
                    "end": utt["end"]
                })
            # Concatenate transcription
            full_text += " " + result.get("text", "")
            language_code = result.get("language_code", language_code)

        return {
            "language": language_code,
            "transcription": full_text.strip(),
            "speakers": all_speakers
        }

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=400, detail="Invalid or corrupted audio file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        if converted_path and os.path.exists(converted_path):
            os.unlink(converted_path)
        for chunk in chunk_paths:
            if os.path.exists(chunk):
                os.unlink(chunk)
