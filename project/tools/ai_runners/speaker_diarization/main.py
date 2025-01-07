import os
import whisperx
import gc
import torch
from typing import Dict, Any
from base_agent import BaseAgent
from agent_registry import AgentRegistry
from logger import debug_logger, error_logger




class SpeakerDiarizationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Speaker Diarization",
            description="Transcribes audio with speaker identification"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        debug_logger.debug(f"Initialized Speaker Diarization agent with device: {self.device}")

    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        if not input_data or 'audio_path' not in input_data:
            return self._handle_error("Audio path required")

        audio_path = os.path.abspath(input_data['audio_path'])
        language = input_data.get('language', 'auto')  # Get language from input_data

        if not os.path.exists(audio_path):
            return self._handle_error(f"Audio file not found at {audio_path}")

        try:
            debug_logger.debug(f"Processing audio file: {audio_path} with language: {language}")
            
            # Load and transcribe audio
            debug_logger.debug("Loading WhisperX model")
            model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
            debug_logger.debug("WhisperX model loaded successfully")
            
            debug_logger.debug(f"Loading audio from path: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            debug_logger.debug("Audio loaded successfully")
            
            debug_logger.debug("Starting transcription")
            # Use selected language if not auto
            transcribe_options = {"batch_size": 16}
            if language != "auto":
                transcribe_options["language"] = language
            result = model.transcribe(audio, **transcribe_options)
            
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            del model

            debug_logger.debug("Starting alignment")
            # Align whisper output
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], 
                device=self.device
            )
            result = whisperx.align(
                result["segments"],
                model_a, 
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )

            # Clear GPU memory again
            gc.collect()
            torch.cuda.empty_cache()
            del model_a

            debug_logger.debug("Starting diarization")
            # Diarize audio
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )

            # Get speaker labels
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            debug_logger.debug("Processing complete")
            return {
                "status": "success",
                "segments": result["segments"],
                "diarize_segments": diarize_segments
            }

        except Exception as e:
            error_logger.error(f"Diarization error: {str(e)}")
            return self._handle_error(str(e))

    def _handle_error(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": message
        }

# Register the agent
AgentRegistry.register(SpeakerDiarizationAgent)

if __name__ == '__main__':
    agent = SpeakerDiarizationAgent()
    agent.run()