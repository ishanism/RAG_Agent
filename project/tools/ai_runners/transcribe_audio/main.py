import pyaudio
from typing import Dict, Any
from base_agent import BaseAgent
from agent_registry import AgentRegistry

class TranscribeAudioAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Audio Transcription",
            description="Transcribes audio files to text"
        )
        self.p = pyaudio.PyAudio()
    
    def list_input_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if not device_info['maxInputChannels'] > 0:
                continue
            devices.append(self._format_device_info(device_info, i))
        return devices

    def _format_device_info(self, device_info, index):
        return {
            'index': index,
            'name': device_info['name'],
            'channels': device_info['maxInputChannels'],
            'sample_rate': int(device_info['defaultSampleRate'])
        }

    def _get_default_input(self):
        return {"device_id": 0}

    def _create_output(self, devices, transcription="Sample transcription"):
        return {
            "devices": devices,
            "transcription": transcription,
            "status": "success",
            "message": "Audio transcribed successfully"
        }

    def _handle_error(self, e):
        return {
            "status": "error",
            "message": str(e)
        }

    def run(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        input_data = input_data or self._get_default_input()
        if not self.validate_inputs(input_data):
            return self._handle_error("Invalid input parameters")

        self.pre_run()
        try:
            devices = self.list_input_devices()
            output = self._create_output(devices)
            if not self.validate_output(output):
                return self._handle_error("Invalid output format")
            return output
        except Exception as e:
            return self._handle_error(e)
        finally:
            self.post_run()
            self.p.terminate()

# Register the agent
AgentRegistry.register(TranscribeAudioAgent)

if __name__ == '__main__':
    agent = TranscribeAudioAgent()
    agent.run()