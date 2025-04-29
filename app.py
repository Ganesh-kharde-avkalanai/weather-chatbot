import streamlit as st
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load and chunk PDF content
def load_all_pdfs_from_directory(directory, chunk_size=200):
    chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Chunk the text
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunks.append(chunk)
    return chunks

# Load PDFs from ./data directory
text_chunks = load_all_pdfs_from_directory("data")

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(text_chunks).astype("float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Gemini setup
genai.configure(api_key="AIzaSyAyXZy3TFg7B2Ai7G6raW9Cep-qzDXkJo8")

def get_gemini_response(user_query):
    try:
        query_embedding = embed_model.encode([user_query]).astype("float32")
        D, I = index.search(query_embedding, k=1)  # top 1 result
        context = text_chunks[I[0][0]]
        print(context)

        prompt = f"""
        system: You are a weather-aware chatbot. Answer the query using the context below:
        
        Context: {context}
        User: {user_query}
        Gemini:
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
st.set_page_config(page_title="RAG Chatbot with FAISS", layout="centered")
st.title("📚 Weather RAG Chatbot with FAISS")

# Session storage for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input prompt + button
if prompt := st.chat_input("Type your message here..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_gemini_response(prompt)
    
    st.session_state.history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
