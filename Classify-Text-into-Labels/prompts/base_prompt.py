"""
Base Prompt class - Base class for all prompts
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional

class TaskResult(BaseModel):
    """Processing result from prompts"""
    success: bool
    result: str
    prompt_used: str
    processing_time: Optional[float] = None
    confidence: Optional[float] = None

class BasePrompt(ABC):
    """Base class for all prompts"""
    def __init__(self, llm):
        self.llm = llm
    @abstractmethod
    async def process(self, user_input: str) -> TaskResult:
        """Process request from user"""
        pass
