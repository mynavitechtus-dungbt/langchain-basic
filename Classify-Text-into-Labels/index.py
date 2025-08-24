"""
AI Application Router – Route user requests to specialized prompts based on classification
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


# Import prompts from individual files
from prompts.customer_support_prompt import CustomerSupportPrompt
from prompts.content_analysis_prompt import ContentAnalysisPrompt
from prompts.data_processing_prompt import DataProcessingPrompt
from prompts.general_chat_prompt import GeneralChatPrompt
from prompts.base_prompt import TaskResult

load_dotenv()

# =================== MODELS ===================

class RequestClassification(BaseModel):
    """Classify the user's request"""
    intent: Literal[
        "customer_support",
        "content_analysis",
        "data_processing",
        "general_chat"
    ] = Field(
        description="The main category of the user's request"
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="The complexity level of the request"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="The urgency level of the request"
    )
    requires_data: bool = Field(
        description="Whether external data access is required"
    )
    confidence: float = Field(
        description="The confidence score of the classification (0.0-1.0)"
    )

# =================== ROUTER CLASS ===================

class AIApplicationRouter:
    def __init__(self):
        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        
        # Router LLM - classify request
        self.router_llm = self.llm.with_structured_output(RequestClassification)
        
        # Specialized Prompts
        self.prompts = {
            "customer_support": CustomerSupportPrompt(self.llm),
            "content_analysis": ContentAnalysisPrompt(self.llm),
            "data_processing": DataProcessingPrompt(self.llm),
            "general_chat": GeneralChatPrompt(self.llm)
        }
        
        self.router_prompt = ChatPromptTemplate.from_template(
            """
            Analyze the user's request and determine the appropriate task type:
            
            User request: {user_input}
            
            Classify accurately to route the request to the appropriate prompt.
            """
        )
    
    async def process_request(self, user_input: str) -> TaskResult:
        """Xử lý request từ user"""
        try:
            # Step 1: Classify the request
            classification = self.router_llm.invoke(
                self.router_prompt.invoke({"user_input": user_input})
            )
            
            # Step 2: Check classification result
            if classification is None:
                return TaskResult(
                    success=False,
                    result="Cannot classify the request. Please check your network or API key.",
                    prompt_used="router"
                )
            
            print(f"🤖 Router detected: {classification.intent} (confidence: {classification.confidence:.2f})")
            
            # Step 3: Check confidence
            if classification.confidence < 0.7:
                return TaskResult(
                    success=False,
                    result="Sorry I can not understand?",
                        prompt_used="router"
                )
            
            # Step 4: Route to the appropriate prompt
            prompt = self.prompts.get(classification.intent)
            if not prompt:
                return TaskResult(
                    success=False,
                    result="No suitable prompt found for this request.",
                    prompt_used="router"
                )
            
            # Step 5: Process the request with the selected prompt
            result = await prompt.process(user_input)
            return result
            
        except Exception as e:
            return TaskResult(
                success=False,
                result=f"Error process_request: {str(e)}",
                    prompt_used="error_handler"
            )

# =================== DEMO ===================

async def main():
    router = AIApplicationRouter()
    
    # Test cases
    test_case = "i want to buy a thermos bottle"
    
    print("=== AI APPLICATION ROUTER DEMO ===\n")
    
    result = await router.process_request(test_case)
        
    print(f"✅ Prompt: {result.prompt_used}")
    print(f"📝 Response: {result.result}")
    print(f"🎯 Success: {result.success}")
        

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
