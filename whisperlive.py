from hashlib import file_digest
import pyaudio
import numpy as np
import torch
import time
import threading
import queue
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import os
import sys
import warnings
import dotenv

dotenv.load_dotenv(".env")  # load .env file 

# --- Configuration ---
# VB-Cable Device Search Term (adjust if necessary)
# Common names: "CABLE Output (VB-Audio Virtual Cable)", "VB-Audio Point" (Input device)
VB_CABLE_NAME_PART = "Out B1" # Or "VB-Audio Point" or similar based on your system

# Whisper Model Configuration
MODEL_SIZE = "large-v3"  # Options: "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
                       # Use ".en" models for English-only meetings for better performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8" # or "float32" / "int8"

# Pyannote Configuration
# Make sure you have accepted the user agreement for these models on Hugging Face Hub
PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# Get your Hugging Face access token from https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") # Recommended: set as environment variable
if not HF_TOKEN:
    # If not set via env var, uncomment the line below and paste your token
    # HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE"
    print("Warning: Hugging Face token not found in environment variables.")
    print("Please set the HUGGINGFACE_TOKEN environment variable or paste it directly into the script.")
    # sys.exit("Hugging Face token is required for pyannote.audio.") # Optional: exit if no token


# Audio Configuration
CHUNK_DURATION_S = 5       # Process audio in X-second chunks
SAMPLE_RATE = 16000        # Whisper standard sample rate
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_S) # Samples per chunk
BUFFER_DURATION_S = CHUNK_DURATION_S # Size of the buffer for processing
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION_S)
FORMAT = pyaudio.paInt16   # Audio format
CHANNELS = 1               # Mono audio

# --- Global Variables ---
audio_buffer = np.array([], dtype=np.int16)
buffer_lock = threading.Lock()
data_queue = queue.Queue()
running = True

# --- Helper Functions ---
def find_device_index(p, name_part):
    """Finds the index of an audio device containing name_part."""
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if (name_part.lower() in device_info.get('name').lower() and
                device_info.get('maxInputChannels')) > 0:
            print(f"Found device: {device_info.get('name')} at index {i}")
            return i
    return None

def audio_callback(in_data, frame_count, time_info, status):
    """Callback function for PyAudio stream."""
    global audio_buffer
    new_data = np.frombuffer(in_data, dtype=np.int16)
    with buffer_lock:
        audio_buffer = np.append(audio_buffer, new_data)
        # Keep buffer size manageable, discard oldest if too large
        # A larger overlap might slightly improve context but increases memory
        if len(audio_buffer) > BUFFER_SIZE * 2: # Keep a bit more than needed
             audio_buffer = audio_buffer[-BUFFER_SIZE:] # Keep the last buffer's worth

        # Check if we have enough data to process
        if len(audio_buffer) >= BUFFER_SIZE:
            # Put a copy of the buffer onto the queue for processing
            data_queue.put(audio_buffer.copy())
            # Clear the buffer after copying
            audio_buffer = np.array([], dtype=np.int16)

    return (in_data, pyaudio.paContinue)

def process_audio_chunk(whisper_model, diarize_pipeline, audio_chunk_np):
    """Transcribes and diarizes a single audio chunk."""
    try:
        # Convert to float32 for models
        audio_float32 = audio_chunk_np.astype(np.float32) / 32768.0

        # 1. Diarization
        #    Need to reshape for pyannote [1, num_samples]
        audio_for_diarize = torch.from_numpy(audio_float32).unsqueeze(0)
        diarization = diarize_pipeline({"waveform": audio_for_diarize, "sample_rate": SAMPLE_RATE})

        # 2. Transcription
        segments, info = whisper_model.transcribe(
            audio_float32,
            beam_size=5,
            language="en", # Assuming English for .en models, otherwise detect
            # word_timestamps=True # Enable if you need word-level timestamps
        )

        # 3. Combine results (simple approach: assign speaker based on max overlap)
        print("-" * 30)
        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            text = segment.text.strip()

            if not text:
                continue

            # Find speaker for this segment
            segment_speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Check for overlap between transcription segment and diarization turn
                overlap_start = max(start_time, turn.start)
                overlap_end = min(end_time, turn.end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0:
                    segment_speakers.append({"speaker": speaker, "duration": overlap_duration})

            # Assign the speaker with the maximum overlap duration
            if segment_speakers:
                dominant_speaker = max(segment_speakers, key=lambda x: x['duration'])['speaker']
            else:
                dominant_speaker = "UNKNOWN" # Handle cases with no speaker overlap

            print(f"[{dominant_speaker}] ({segment.start:.2f}s -> {segment.end:.2f}s): {text}")

    except Exception as e:
        print(f"Error processing chunk: {e}", file=sys.stderr)

def main_processing_loop(whisper_model, diarize_pipeline):
    """Main loop to fetch data from queue and process it."""
    while running or not data_queue.empty():
        try:
            # Get data from the queue, wait up to 1 second
            audio_chunk = data_queue.get(timeout=1)
            process_audio_chunk(whisper_model, diarize_pipeline, audio_chunk)
            data_queue.task_done() # Mark task as done
        except queue.Empty:
            # If the queue is empty and we are no longer running, break
            if not running:
                break
            continue # Continue waiting if running
        except Exception as e:
            print(f"Error in processing loop: {e}", file=sys.stderr)
            # Optionally add a small sleep to prevent tight error loops
            time.sleep(0.1)

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing...")

    # Suppress harmless warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.functional')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Find VB-Cable device
    input_device_index = find_device_index(p, VB_CABLE_NAME_PART)
    if input_device_index is None:
        print(f"Error: Could not find VB-Cable input device containing '{VB_CABLE_NAME_PART}'.")
        print("Available input devices:")
        for i in range(p.get_device_count()):
             dev = p.get_device_info_by_index(i)
             if int(dev['maxInputChannels']) > 0:
                  print(f"  {i}: {dev['name']} (Channels: {dev['maxInputChannels']})")
        p.terminate()
        sys.exit(1)

    # Load Faster Whisper model
    print(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...")
    try:
        whisper = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Whisper model loaded.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}", file=sys.stderr)
        p.terminate()
        sys.exit(1)

    # Load Pyannote Diarization pipeline
    print("Loading Pyannote diarization pipeline...")
    try:
        if not HF_TOKEN:
             print("Warning: Proceeding without Hugging Face token. Model download might fail if not cached or agreements not accepted.")
             print("Get a token at https://huggingface.co/settings/tokens")
             diarization_pipeline = Pipeline.from_pretrained(
                PYANNOTE_DIARIZATION_MODEL,
                # No token passed, relies on cached model or prior login
             )
        else:
             diarization_pipeline = Pipeline.from_pretrained(
                PYANNOTE_DIARIZATION_MODEL,
                use_auth_token=HF_TOKEN
             )
        # Send pipeline to GPU if available
        if DEVICE == "cuda":
            diarization_pipeline.to(torch.device("cuda"))
        print("Pyannote pipeline loaded.")
    except Exception as e:
        print(f"Error loading Pyannote pipeline: {e}", file=sys.stderr)
        print("Ensure you have accepted model terms on Hugging Face and have a valid token if needed.")
        p.terminate()
        sys.exit(1)


    # Start audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE, # Read slightly larger chunks for callback efficiency
                    input_device_index=input_device_index,
                    stream_callback=audio_callback)

    print("Starting audio stream... Press Ctrl+C to stop.")
    stream.start_stream()

    # Start the processing thread
    processing_thread = threading.Thread(target=main_processing_loop, args=(whisper, diarization_pipeline))
    processing_thread.start()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        running = False # Signal threads to stop
    finally:
        # Wait for the processing thread to finish remaining tasks
        processing_thread.join(timeout=CHUNK_DURATION_S + 2) # Wait a bit longer than chunk duration
        if processing_thread.is_alive():
             print("Warning: Processing thread did not finish gracefully.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio stream stopped.")