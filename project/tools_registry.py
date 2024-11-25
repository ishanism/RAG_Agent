
from typing import Dict, Type
from base_tool import BaseTool

class ToolRegistry:
    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> None:
        """Register a new tool class"""
        cls._tools[tool_class.__name__] = tool_class
        
    @classmethod
    def get_tool(cls, tool_name: str) -> Type[BaseTool]:
        """Get tool class by name"""
        return cls._tools.get(tool_name)
    
    @classmethod
    def get_all_tools(cls) -> Dict[str, Type[BaseTool]]:
        """Get all registered tools"""
        return cls._tools