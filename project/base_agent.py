import yaml
import jsonschema
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from langchain_core.callbacks import CallbackManager  # Updated import path
from logger import debug_logger, info_logger, warning_logger, error_logger

class BaseAgent(ABC):
    """Base class for all agents and tools"""
    
    def __init__(
        self,
        name: str,
        description: str,
        callback_manager: Optional[CallbackManager] = None,  # Fixed bracket syntax here
        **kwargs
    ):
        self.name = name
        self.description = description
        self.callback_manager = callback_manager
        self.kwargs = kwargs
        info_logger.info(f"Initialized {self.name} agent")
        self._load_schema()

    def _load_schema(self):
        """Load schema for this agent"""
        with open('schema.yml', 'r') as f:
            schemas = yaml.safe_load(f)
            agent_schema = schemas['schemas'].get(self.__class__.__name__, {})
            self.input_schema = agent_schema.get('input', {})
            self.output_schema = agent_schema.get('output', {})

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Main execution method that must be implemented by all agents"""
        pass

    def validate_inputs(self, input_data: Dict) -> bool:
        """Validate input data against schema"""
        try:
            jsonschema.validate(instance=input_data, schema=self.input_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            error_logger.error(f"Input validation error: {str(e)}")
            return False

    def validate_output(self, output_data: Dict) -> bool:
        """Validate output data against schema"""
        try:
            jsonschema.validate(instance=output_data, schema=self.output_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            error_logger.error(f"Output validation error: {str(e)}")
            return False

    def pre_run(self) -> None:
        """Hook for operations before running the agent"""
        debug_logger.debug(f"Starting {self.name} execution")

    def post_run(self) -> None:
        """Hook for cleanup operations after running the agent"""
        debug_logger.debug(f"Completed {self.name} execution")

    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }