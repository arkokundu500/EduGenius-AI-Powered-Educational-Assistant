import streamlit as st
import google.generativeai as genai
import os
import re
from pathlib import Path
from collections import defaultdict


CUSTOM_DATA_DIR = "public"  
MAX_CONTEXT_LENGTH = 30000  
CHUNK_SIZE = 500  

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')

def load_custom_data():
    """Load and preprocess all text files with improved chunking"""
    custom_data = []
    data_dir = Path(CUSTOM_DATA_DIR)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Custom data directory '{CUSTOM_DATA_DIR}' not found")
    
    for txt_file in data_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    words = sentence.split()
                    if current_length + len(words) <= CHUNK_SIZE:
                        current_chunk.append(sentence)
                        current_length += len(words)
                    else:
                        custom_data.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(words)
                
                if current_chunk:
                    custom_data.append(" ".join(current_chunk))
    
    return custom_data

def find_relevant_chunks(query, custom_data):
    """Improved relevance scoring with term frequency"""
    query = query.lower()
    query_terms = re.findall(r'\b\w+\b', query)
    term_weights = defaultdict(int)
    
    
    for term in query_terms:
        term_weights[term] += 1
    
    chunk_scores = []
    for chunk in custom_data:
        chunk_lower = chunk.lower()
        score = 0
        for term, weight in term_weights.items():
            if term in chunk_lower:
        
                score += weight * 2
                
                if f" {term}" in chunk_lower or f"{term} " in chunk_lower:
                    score += weight
        chunk_scores.append((score, chunk))
    
    
    chunk_scores.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in chunk_scores if score > 0][:7]  

def generate_response(query, custom_data):
    """Enhanced response generation with context prioritization"""
    relevant_chunks = find_relevant_chunks(query, custom_data)
    
    if not relevant_chunks:
        
        return model.generate_content(
            f"Answer this question: {query} "
            "(Note: No relevant information found in provided materials)"
        ).text
    
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are an educational assistant. Follow these steps:
    1. FIRST analyze this context from provided materials:
    {context}
    2. If the answer exists in the context, use it verbatim where possible
    3. If needed, supplement with your knowledge but clearly state what's from context vs general knowledge
    4. If conflicting info exists, prioritize the context
    5. Never mention you're an AI or language model
    6. Provide explanations and reasoning where necessary
    
    Question: {query}
    Answer:"""
    
    if len(prompt) > MAX_CONTEXT_LENGTH:
        prompt = prompt[:MAX_CONTEXT_LENGTH]
    
    return model.generate_content(prompt).text

try:
    custom_data = load_custom_data()
    if not custom_data:
        raise ValueError(f"No valid data found in {CUSTOM_DATA_DIR}")
except Exception as e:
    st.error(f"Data loading error: {str(e)}")
    st.stop()

st.set_page_config(page_title="VLabs- Assistant AI", page_icon="🎓")
st.title(" Education Assistant with VLABS Knowledge")
st.caption(f"Provide your queries related to educational content")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Revealing answers in few seconds..."):
        try:
            response = generate_response(prompt, custom_data)
        except Exception as e:
            response = f"Error: {str(e)}"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})