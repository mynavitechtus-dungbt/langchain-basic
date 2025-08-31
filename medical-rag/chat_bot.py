import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt




def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know "i don't know.", dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                print("[DEBUG] Vectorstore is None")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.0,
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response.get("result", "[No result]")
            source_documents = response.get("source_documents", [])
            sources = set()
            for doc in source_documents:
                file_name = doc.metadata.get('source', '[unknown]')
                page = doc.metadata.get('page', '[unknown]')
                sources.add(f"{file_name} (page {page})")
            sources_str = ', '.join(sources)
            result_to_show = result
 
            if sources and result.strip().lower() != "i don't know.":
                result_to_show += f"\n\nSources: {sources_str}"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()