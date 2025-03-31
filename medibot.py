import os
import streamlit as st
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import random

# Path to FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

# API Config (Example: OpenFDA)
MEDICAL_API_URL = "https://api.fda.gov/drug/event.json"
API_PARAMS = {"limit": 1}  # Adjust this based on API limits

@st.cache_resource
def get_vectorstore():
    """Lazy load FAISS vector store only when needed."""
    if not os.path.exists(DB_FAISS_PATH):
        st.error("FAISS veritabanı yolu bulunamadı. Lütfen dizini kontrol edin.")
        return None
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS: {e}")
        return None

def set_custom_prompt():
    """Create a custom prompt for retrieval-based answers."""
    return PromptTemplate(template=""" 
        Use the provided context to answer the user's medical question accurately.
        If you don't know the answer, just say "I don't know." Don't make up an answer.

        Context: {context}
        Question: {question}

        Provide a structured and clear answer.
    """, input_variables=["context", "question"])

def load_llm():
    """Load the Hugging Face model for retrieval-based Q&A."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("Hugging Face API anahtarı bulunamadı! Lütfen çevresel değişkenleri kontrol edin.")
        return None
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

def get_medical_data(user_query):
    """Fetch real-time health data from an external medical API."""
    try:
        response = requests.get(MEDICAL_API_URL, params=API_PARAMS)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                return f"📡 **Live Medical Data:** {data['results'][0].get('summaryReport', 'No summary available')}"
        return None
    except Exception as e:
        return f"⚠ Error fetching live data: {str(e)}"

def get_general_response(user_input):
    """Handles general conversation with improved responses."""
    general_chat_responses = {
        "hello": ["Hello! 😊 How can I help you today?", "Hey there! Need any help?", "Hi! What's up?"],
        "hi": ["Hi! How’s your day going?", "Hey! What can I do for you?", "Hello! Need any info?"],
        "how are you": ["I’m just a bot, but thanks for asking! 😊", "I'm great! How about you?", "Doing fine! Ready to assist you!"],
        "who are you": ["I’m MediBot, your AI assistant for medical queries!", "I'm MediBot, an AI chatbot trained for medical questions."],
        "thank you": ["You're welcome! 😊", "No problem! Happy to help!", "Glad I could assist!"],
        "bye": ["Goodbye! Have a great day! 😊", "Bye! Take care!", "See you next time!"]
    }
    for key, responses in general_chat_responses.items():
        if key in user_input.lower():
            return random.choice(responses)
    return None

def format_citation(result):
    """Format the citation for the medical data."""
    citation = "Source: Hamilton W, Peters TJ, Round A, et al. 'Clinical features of lung cancer.' p. 1060."
    return f"{result}\n\n{citation}"

def main():
    """Main Streamlit application."""
    st.title("Ask MediBot! 🤖 Your AI Medical Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Ask a medical question or just say hello!")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
    
        general_response = get_general_response(prompt)
        if general_response:
            st.chat_message('assistant').markdown(general_response)
            st.session_state.messages.append({'role': 'assistant', 'content': general_response})
            return
    
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return
    
        llm = load_llm()
        if llm is None:
            return
    
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )
    
            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "I don't know")
    
            if not result or result.lower() == "i don't know":
                live_data = get_medical_data(prompt)
                if live_data:
                    result = live_data
    
            formatted_result = format_citation(result)
            st.chat_message('assistant').markdown(formatted_result)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_result})
    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()