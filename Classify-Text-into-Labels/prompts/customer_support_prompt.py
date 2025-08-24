"""
Customer Support Prompt - Customer support specialist
"""

from langchain_core.prompts import ChatPromptTemplate
from .base_prompt import BasePrompt, TaskResult

class CustomerSupportPrompt(BasePrompt):
    """Prompt specialized in customer support"""
    def __init__(self, llm):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a customer support specialist. Respond professionally, 
            friendly and solve problems effectively.
            
            Customer issue: {user_input}
            
            Provide specific solutions and detailed guidance.
            """
        )
    async def process(self, user_input: str) -> TaskResult:
        """Process customer support requests"""
        response = self.llm.invoke(self.prompt.invoke({"user_input": user_input}))
        return TaskResult(
            success=True,
            result=response.content,
            prompt_used="customer_support",
            confidence=0.95
        )
