"""
Content Analysis Prompt - Content analysis specialist
"""

from langchain_core.prompts import ChatPromptTemplate
from .base_prompt import BasePrompt, TaskResult

class ContentAnalysisPrompt(BasePrompt):
    """Prompt specialized in content analysis"""
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a content analysis specialist. Analyze deeply and provide 
            valuable insights about the provided content.
            
            Content to analyze: {user_input}
            
            Analyze: sentiment, topics, purpose, target audience.
            """
        )
    async def process(self, user_input: str) -> TaskResult:
        """Process content analysis requests"""
        response = self.llm.invoke(self.prompt.invoke({"user_input": user_input}))
        return TaskResult(
            success=True,
            result=response.content,
            prompt_used="content_analysis",
            confidence=0.92
        )
