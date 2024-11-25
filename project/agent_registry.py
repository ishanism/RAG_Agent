
from typing import Dict, Type
from base_agent import BaseAgent

class AgentRegistry:
    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> None:
        """Register a new agent class"""
        cls._agents[agent_class.__name__] = agent_class
        
    @classmethod
    def get_agent(cls, agent_name: str) -> Type[BaseAgent]:
        """Get agent class by name"""
        return cls._agents.get(agent_name)
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, Type[BaseAgent]]:
        """Get all registered agents"""
        return cls._agents