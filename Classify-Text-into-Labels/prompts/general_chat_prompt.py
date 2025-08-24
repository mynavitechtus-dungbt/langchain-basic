"""
General Chat Prompt - General AI assistant
"""

from langchain_core.prompts import ChatPromptTemplate
from .base_prompt import BasePrompt, TaskResult

class GeneralChatPrompt(BasePrompt):
    """Prompt for general conversation"""
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a friendly and helpful AI assistant. Chat naturally 
            and provide accurate information.
            
            User message: {user_input}
            
            Respond in a friendly and helpful manner.
            """
        )
    async def process(self, user_input: str) -> TaskResult:
        """Process general conversation"""
        response = self.llm.invoke(self.prompt.invoke({"user_input": user_input}))
        return TaskResult(
            success=True,
            result=response.content,
            prompt_used="general_chat",
            confidence=0.85
        )
