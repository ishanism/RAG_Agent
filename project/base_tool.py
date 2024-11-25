
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import yaml
import jsonschema
from logger import debug_logger, info_logger, error_logger

class BaseTool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        **kwargs
    ):
        self.name = name
        self.description = description
        self.kwargs = kwargs
        info_logger.info(f"Initialized {self.name} tool")
        self._load_schema()

    def _load_schema(self):
        with open('schema.yml', 'r') as f:
            schemas = yaml.safe_load(f)
            tool_schema = schemas['schemas'].get(self.__class__.__name__, {})
            self.input_schema = tool_schema.get('input', {})
            self.output_schema = tool_schema.get('output', {})

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    def validate_inputs(self, input_data: Dict) -> bool:
        try:
            jsonschema.validate(instance=input_data, schema=self.input_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            error_logger.error(f"Tool input validation error: {str(e)}")
            return False

    def validate_output(self, output_data: Dict) -> bool:
        try:
            jsonschema.validate(instance=output_data, schema=self.output_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            error_logger.error(f"Tool output validation error: {str(e)}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__
        }