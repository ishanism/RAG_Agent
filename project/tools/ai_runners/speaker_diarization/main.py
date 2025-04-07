import os
from faster_whisper import download_model
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

    def _serialize_segments(self, segments, diarize_segments):
        """Convert segments and diarization data to JSON-serializable format"""
        debug_logger.debug(f"Serializing {len(segments)} segments")
        try:
            serialized_segments = []
            for i, segment in enumerate(segments):
                debug_logger.debug(f"Serializing segment {i+1}/{len(segments)}")
                serialized_segment = {
                    "start": float(segment.get("start", 0)),
                    "end": float(segment.get("end", 0)),
                    "text": str(segment.get("text", "")),
                    "speaker": str(segment.get("speaker", "UNKNOWN"))
                }
                serialized_segments.append(serialized_segment)

            debug_logger.debug("Serialization complete")
            return serialized_segments
        except Exception as e:
            error_logger.error(f"Serialization error: {str(e)}", exc_info=True)
            raise

    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        debug_logger.debug(f"Run called with input: {input_data}")
        
        if not input_data or 'audio_path' not in input_data:
            debug_logger.error("Missing audio_path in input")
            return self._handle_error("Audio path required")

        audio_path = os.path.abspath(input_data['audio_path'])
        language = input_data.get('language', 'auto')
        debug_logger.debug(f"Processing with language: {language}")

        if not os.path.exists(audio_path):
            return self._handle_error(f"Audio file not found at {audio_path}")

        try:
            debug_logger.debug(f"Processing audio file: {audio_path} with language: {language}")
            
            debug_logger.debug(f"Model device: {self.device}, compute_type: {self.compute_type}")
            debug_logger.debug("Loading WhisperX model")
            model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type, )
            debug_logger.debug("WhisperX model loaded successfully")
            
            debug_logger.debug(f"Loading audio from path: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            debug_logger.debug("Audio loaded successfully")
            
            debug_logger.debug("Starting transcription")
            # Use selected language if not auto
            transcribe_options = {"batch_size": 16}
            if language != "auto":
                transcribe_options["language"] = language
            
            debug_logger.debug("Starting transcription with options:")
            debug_logger.debug(f"Transcription options: {transcribe_options}")
            result = model.transcribe(audio, **transcribe_options)
            debug_logger.debug(f"Detected language: {result.get('language')}")
            
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

            debug_logger.debug("Starting diarization with HF token")
            if not self.hf_token:
                debug_logger.warning("No Hugging Face token found")
            # Diarize audio
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )

            # Get speaker labels
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Before returning, serialize the data
            serialized_result = self._serialize_segments(result["segments"], diarize_segments)
            
            debug_logger.debug(f"Final result contains {len(serialized_result)} segments")
            debug_logger.debug("Processing complete")
            return {
                "status": "success",
                "segments": serialized_result
            }

        except Exception as e:
            error_logger.error(f"Diarization error: {str(e)}", exc_info=True)
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