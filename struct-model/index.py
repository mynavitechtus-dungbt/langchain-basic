from langchain.chat_models import init_chat_model
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Union

# Load environment variables from .env file
load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


structured_llm = llm.with_structured_output(Joke, include_raw=True)


print('====================response_ai_with_struct=========================')
response_ai_with_struct = structured_llm.invoke("Tell me a joke about cats and rate it from 1 to 10")
print(response_ai_with_struct["parsed"].setup)      # "What do you call a cat that can swim?"
print(response_ai_with_struct["parsed"].punchline)  # "A catfish!"
print(response_ai_with_struct["parsed"].rating)



