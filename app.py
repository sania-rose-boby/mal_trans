from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
import whisper
import numpy as np
import torch
import tempfile
import os
import wave
import base64
import time
from transformers import MarianMTModel, MarianTokenizer
import contextlib
import re

app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Whisper and MarianMT models...")
whisper_model = whisper.load_model("base", device=device)
model_name = "Helsinki-NLP/opus-mt-en-ml"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name).to(device)
print("Models loaded successfully.")

audio_buffer = []
sample_rate = 16000
buffer_duration_sec = 3
buffer_size = buffer_duration_sec * sample_rate

AUDIO_DIR = "stored_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Constants for text processing
MAX_TEXT_LENGTH = 100000  # Maximum characters allowed
MAX_CHUNK_LENGTH = 800    # Reduced for better context
MIN_CHUNK_LENGTH = 50     # Minimum chunk size to process
CONTEXT_OVERLAP = 100     # Number of characters to overlap between chunks

def validate_text(text):
    """Validate text input and return any issues"""
    if not text or not isinstance(text, str):
        return "No text provided"
    
    if len(text.strip()) == 0:
        return "Empty text provided"
    
    if len(text) > MAX_TEXT_LENGTH:
        return f"Text too long. Maximum {MAX_TEXT_LENGTH} characters allowed"
    
    return None

def post_process_malayalam(text):
    """Post-process Malayalam translation for better quality"""
    # Fix common translation issues
    replacements = {
        'വിനിമയതലം': 'വിനിമയം',  # Common incorrect translation
        'ശബ്ദങ്ങൾ': 'ശബ്ദം',      # Fix pluralization
        'സാംസ് കാരിക': 'സാമൂഹിക',  # Fix common typos
        'സാംസ്കാരികവും': 'സാംസ്കാരിക',  # Fix repeated words
        'സ്വകാര്യത, സ്വകാര്യത': 'സ്വകാര്യത',  # Remove duplicates
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'([.,!?])\s*', r'\1 ', text)  # Add space after punctuation
    
    # Fix common sentence structure issues
    text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    
    return text.strip()

def split_text(text, max_length=MAX_CHUNK_LENGTH):
    """
    Enhanced text splitting function with context preservation
    """
    # Validate input
    error = validate_text(text)
    if error:
        return {"error": error, "chunks": [], "total_chunks": 0}
    
    # Normalize whitespace and line endings
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ''
    total_chunks = 0
    previous_context = ''
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If paragraph is too long, split it into sentences
        if len(paragraph) > max_length:
            # Improved sentence splitting: handles all punctuation, multiple spaces, and ensures all text is included
            sentences = re.findall(r'[^.!?;:]+[.!?;:]?', paragraph)
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Add previous context if available
                if previous_context and len(current_chunk) < CONTEXT_OVERLAP:
                    current_chunk = previous_context + ' ' + current_chunk
                
                # Check if adding this sentence would exceed max_length
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += (' ' + sentence if current_chunk else sentence)
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk)
                        total_chunks += 1
                        # Save last part of chunk as context for next chunk
                        previous_context = ' '.join(current_chunk.split()[-5:])
                    current_chunk = sentence
        else:
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                current_chunk += ('\n\n' + paragraph if current_chunk else paragraph)
            else:
                # Save current chunk and start new one with paragraph
                if current_chunk:
                    chunks.append(current_chunk)
                    total_chunks += 1
                    previous_context = ' '.join(current_chunk.split()[-5:])
                current_chunk = paragraph
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)
        total_chunks += 1
    
    return {
        "chunks": chunks,
        "total_chunks": total_chunks,
        "total_characters": len(text),
        "average_chunk_size": len(text) / total_chunks if total_chunks > 0 else 0
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate_text():
    try:
        data = request.get_json()
        text = data.get("text", "")
        
        # Validate text
        error = validate_text(text)
        if error:
            return {"error": error}, 400

        # Split text into chunks with progress information
        split_result = split_text(text)
        if "error" in split_result:
            return {"error": split_result["error"]}, 400
            
        chunks = split_result["chunks"]
        total_chunks = split_result["total_chunks"]
        
        print(f"[DEBUG] Processing {total_chunks} chunks for translation")
        
        translations = []
        for i, chunk in enumerate(chunks, 1):
            try:
                # Add special handling for technical terms
                chunk = chunk.replace('technology', 'സാങ്കേതികവിദ്യ')
                chunk = chunk.replace('communication', 'ആശയവിനിമയം')
                
                inputs = tokenizer([chunk], return_tensors="pt", padding=True).to(device)
                translated = translator.generate(**inputs, max_length=512, num_beams=4)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                
                # Post-process the translation
                translated_text = post_process_malayalam(translated_text)
                
                translations.append(translated_text)
                print(f"[DEBUG] Translated chunk {i}/{total_chunks}")
            except Exception as e:
                print(f"[DEBUG] Error translating chunk {i}: {str(e)}")
                translations.append(f"[Translation error in chunk {i}]")
        
        # Combine translations with proper spacing
        full_translation = ' '.join(translations)
        full_translation = post_process_malayalam(full_translation)
        
        return {
            "translated_text": full_translation,
            "stats": {
                "total_chunks": total_chunks,
                "total_characters": split_result["total_characters"],
                "average_chunk_size": split_result["average_chunk_size"]
            }
        }

    except Exception as e:
        print(f"[DEBUG] Translation error: {str(e)}")
        return {"error": str(e)}, 500

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    try:
        if "audio" not in request.files:
            return {"error": "No audio file provided."}, 400

        file = request.files["audio"]
        if file.filename == "":
            return {"error": "No selected file."}, 400

        UPLOAD_FOLDER = "uploads"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Transcribe with Whisper
        result = whisper_model.transcribe(filepath, fp16=False, language="en")
        print(f"[DEBUG] Uploaded audio Whisper result: {result}")
        english_text = result["text"].strip()

        if english_text:
            inputs = tokenizer([english_text], return_tensors="pt", padding=True).to(device)
            translated = translator.generate(**inputs)
            malayalam_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            malayalam_text = ""

        if os.path.exists(filepath):
            os.remove(filepath)

        return {
            "english": english_text,
            "malayalam": malayalam_text
        }
    except Exception as e:
        print(f"Error in upload_audio: {e}") # Debug print to console
        return {"error": str(e)}, 500 # Return JSON error to frontend

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

# For real-time audio: buffer all audio, and on stop, process the full buffer
user_audio_buffers = {}
user_audio_partial_index = {}

@socketio.on("connect")
def handle_connect():
    emit("status", {"message": "Connected to server"})

@socketio.on("start_recording")
def handle_start():
    user_audio_buffers[request.sid] = []
    user_audio_partial_index[request.sid] = 0
    emit("recording_started")

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    try:
        audio_base64 = data["audio"]
        if not audio_base64 or len(audio_base64) < 10:
            print("[DEBUG] Skipping empty or too short audio chunk.")
            return
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            emit("error", {"message": f"Base64 decode error: {str(e)}"})
            return
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if request.sid in user_audio_buffers:
            user_audio_buffers[request.sid].extend(audio_np)
            # Only process new audio since last partial
            start_idx = user_audio_partial_index[request.sid]
            while len(user_audio_buffers[request.sid]) - start_idx >= buffer_size:
                print(f"[DEBUG] Partial: start_idx={start_idx}, next_idx={start_idx+buffer_size}, buffer_len={len(user_audio_buffers[request.sid])}")
                segment = np.array(user_audio_buffers[request.sid][start_idx:start_idx+buffer_size])
                user_audio_partial_index[request.sid] += buffer_size
                print(f"[DEBUG] Updated user_audio_partial_index={user_audio_partial_index[request.sid]}")
                start_idx = user_audio_partial_index[request.sid]
                print(f"[DEBUG] Partial segment length: {len(segment)} samples")
                if len(segment) > sample_rate:  # Only emit if at least 1 second
                    audio_filename = f"partial_{request.sid}_{int(time.time()*1000)}.wav"
                    audio_path = os.path.join(AUDIO_DIR, audio_filename)
                    with contextlib.closing(wave.open(audio_path, "wb")) as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        int16_audio = (segment * 32767).astype(np.int16)
                        wf.writeframes(int16_audio.tobytes())
                    print(f"[DEBUG] Saved partial audio to: {audio_path}")
                    result = whisper_model.transcribe(audio_path, fp16=False, language="en")
                    english_text = result["text"].strip()
                    if english_text:
                        inputs = tokenizer([english_text], return_tensors="pt", padding=True).to(device)
                        translated = translator.generate(**inputs)
                        malayalam_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                    else:
                        malayalam_text = ""
                    emit("transcription_result", {
                        "english": english_text,
                        "malayalam": malayalam_text,
                        "timestamp": time.time(),
                        "audio_filename": audio_filename
                    })
    except Exception as e:
        emit("error", {"message": f"Error processing audio chunk: {str(e)}"})

@socketio.on("stop_recording")
def handle_stop():
    try:
        audio_buffer = user_audio_buffers.get(request.sid, [])
        print(f"[DEBUG] Final buffer length: {len(audio_buffer)} samples")
        if not audio_buffer or len(audio_buffer) < sample_rate//2:
            emit("error", {"message": "No audio or too short."})
            return
        # Optionally, emit a final partial for any leftover audio
        start_idx = user_audio_partial_index.get(request.sid, 0)
        if len(audio_buffer) > start_idx:
            segment = np.array(audio_buffer[start_idx:])
            if len(segment) > sample_rate//2:  # Only if it's not too short
                audio_filename = f"partial_{request.sid}_{int(time.time()*1000)}_last.wav"
                audio_path = os.path.join(AUDIO_DIR, audio_filename)
                with contextlib.closing(wave.open(audio_path, "wb")) as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    int16_audio = (segment * 32767).astype(np.int16)
                    wf.writeframes(int16_audio.tobytes())
                print(f"[DEBUG] Saved last partial audio to: {audio_path}")
                result = whisper_model.transcribe(audio_path, fp16=False, language="en")
                english_text = result["text"].strip()
                if english_text:
                    inputs = tokenizer([english_text], return_tensors="pt", padding=True).to(device)
                    translated = translator.generate(**inputs)
                    malayalam_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                else:
                    malayalam_text = ""
                emit("transcription_result", {
                    "english": english_text,
                    "malayalam": malayalam_text,
                    "timestamp": time.time(),
                    "audio_filename": audio_filename
                })
        # Final combined result
        segment = np.array(audio_buffer)
        audio_filename = f"final_{request.sid}_{int(time.time()*1000)}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        with contextlib.closing(wave.open(audio_path, "wb")) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int16_audio = (segment * 32767).astype(np.int16)
            wf.writeframes(int16_audio.tobytes())
        result = whisper_model.transcribe(audio_path, fp16=False, language="en")
        print(f"[DEBUG] Whisper transcription result: {result}")
        english_text = result["text"].strip()
        if english_text:
            inputs = tokenizer([english_text], return_tensors="pt", padding=True).to(device)
            translated = translator.generate(**inputs)
            malayalam_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            malayalam_text = ""
        emit("final_transcription_result", {
            "english": english_text,
            "malayalam": malayalam_text,
            "timestamp": time.time(),
            "audio_filename": audio_filename
        })
        user_audio_buffers.pop(request.sid, None)
        user_audio_partial_index.pop(request.sid, None)
    except Exception as e:
        emit("error", {"message": f"Error processing final audio: {str(e)}"})
        user_audio_buffers.pop(request.sid, None)
        user_audio_partial_index.pop(request.sid, None)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)