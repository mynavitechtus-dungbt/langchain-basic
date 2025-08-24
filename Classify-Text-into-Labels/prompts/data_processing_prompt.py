"""
Data Processing Prompt - Data processing specialist
"""

from langchain_core.prompts import ChatPromptTemplate
from .base_prompt import BasePrompt, TaskResult

class DataProcessingPrompt(BasePrompt):
    """Prompt specialized in data processing"""
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a data processing specialist. Provide specific guidance 
            for processing, analyzing, or transforming data.
            
            Data processing request: {user_input}
            
            Provide code examples and step-by-step explanations.
            """
        )
    async def process(self, user_input: str) -> TaskResult:
        """Process data-related requests"""
        response = self.llm.invoke(self.prompt.invoke({"user_input": user_input}))
        return TaskResult(
            success=True,
            result=response.content,
            prompt_used="data_processing",
            confidence=0.88
        )
