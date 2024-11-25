from base_agent import BaseAgent
from agent_registry import AgentRegistry
from typing import Dict, Any

class LiveCaptionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Live Caption",
            description="Generates real-time captions from audio input"
        )
    
    def _get_default_input(self):
        return {
            "device_id": 0,
            "language": "en"
        }

    def _create_output(self, caption="Sample caption"):
        return {
            "caption": caption,
            "status": "success",
            "message": "Captions generated successfully"
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
            output = self._create_output()
            if not self.validate_output(output):
                return self._handle_error("Invalid output format")
            return output
        except Exception as e:
            return self._handle_error(e)
        finally:
            self.post_run()

# Register the agent
AgentRegistry.register(LiveCaptionAgent)

if __name__ == '__main__':
    agent = LiveCaptionAgent()
    agent.run()