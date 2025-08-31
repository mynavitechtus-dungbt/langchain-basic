import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# Load vectorstore
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer. 
    Dont provide anything out of the given context

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    ),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

query = "What is cancer?"
response = qa_chain.invoke({'query': query})
print("Result:", response["result"])
print("Source Documents:", response["source_documents"])
