import streamlit as st
import google.generativeai as genai
import os
import re
from pathlib import Path
from collections import defaultdict
import io  
import time 
from pypdf import PdfReader  
import docx  


CUSTOM_DATA_DIR = "public"  
MAX_CONTEXT_LENGTH = 30000  
CHUNK_SIZE = 500  

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')

# --- FIX: Initialize session state for processed files ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(file_bytes):
    """Extracts text from a DOCX file."""
    try:
        doc_file = io.BytesIO(file_bytes)
        doc = docx.Document(doc_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

def save_text_to_public(text_content, original_filename):
    """Saves extracted text to a new file in the public directory."""
    try:
        
        base_name = Path(original_filename).stem
        timestamp = int(time.time())
        new_filename = f"{base_name}_{timestamp}.txt"
        
        save_path = Path(CUSTOM_DATA_DIR) / new_filename
        
        
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return new_filename
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None



@st.cache_data(ttl=600) 
def load_custom_data():
    """Load and preprocess all text files with improved chunking"""
    custom_data = []
    data_dir = Path(CUSTOM_DATA_DIR)
    
    if not data_dir.exists():
        st.warning(f"Custom data directory '{CUSTOM_DATA_DIR}' not found. Creating it.")
        data_dir.mkdir(exist_ok=True)
    
    for txt_file in data_dir.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        words = sentence.split()
                        if not words:
                            continue

                        if current_length + len(words) <= CHUNK_SIZE:
                            current_chunk.append(sentence)
                            current_length += len(words)
                        else:
                            if current_chunk: # Save the previous chunk
                                custom_data.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_length = len(words)
                    
                    if current_chunk: # Add the last remaining chunk
                        custom_data.append(" ".join(current_chunk))
        except Exception as e:
            st.error(f"Error reading file {txt_file.name}: {e}")
    
    if not custom_data:
        st.info("No text files found in the 'public' directory yet. Upload some documents in the sidebar to get started!")

    return custom_data

def find_relevant_chunks(query, custom_data):
    """Improved relevance scoring with term frequency"""
    query = query.lower()
    query_terms = re.findall(r'\b\w+\b', query)
    
    if not query_terms:
        return []
    
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
            "(Note: No relevant information found in provided materials. You can upload documents in the sidebar.)"
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

# --- Main App Logic ---

st.set_page_config(page_title="VLabs- Assistant AI", page_icon="🎓")
st.title(" EduGenius - Your Constant Learning Assistant")
st.caption(f"Either Upload documents or provide your queries related to educational content")

# --- Sidebar for File Upload (with Loop Fix) ---
with st.sidebar:
    st.header("Add Knowledge")
    st.markdown("Upload PDF or Word documents to add them to the knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Upload files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        files_processed_this_run = False
        new_files_to_process = []

        # --- FIX: Check which files are new ---
        for uploaded_file in uploaded_files:
            if uploaded_file.file_id not in st.session_state.processed_files:
                new_files_to_process.append(uploaded_file)

        if new_files_to_process:
            for uploaded_file in new_files_to_process:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_bytes = uploaded_file.getvalue()
                    text_content = None
                    
                    if uploaded_file.type == "application/pdf":
                        text_content = extract_text_from_pdf(file_bytes)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        text_content = extract_text_from_docx(file_bytes)
                    
                    if text_content:
                        saved_filename = save_text_to_public(text_content, uploaded_file.name)
                        if saved_filename:
                            st.success(f"Added '{uploaded_file.name}' as '{saved_filename}'")
                            # --- FIX: Add file ID to processed list ---
                            st.session_state.processed_files.append(uploaded_file.file_id)
                            files_processed_this_run = True
                        else:
                            st.error(f"Failed to save '{uploaded_file.name}'")
                    else:
                        st.error(f"Could not extract text from '{uploaded_file.name}'")
            
            # --- FIX: Only rerun if new files were processed ---
            if files_processed_this_run:
                st.cache_data.clear()
                st.rerun()

    if st.button("Reload Knowledge Base"):
        # Clear processed file list and cache, then rerun
        st.session_state.processed_files = [] 
        st.cache_data.clear()
        st.rerun()
# --- End of Sidebar ---


try:
    custom_data = load_custom_data()
except Exception as e:
    st.error(f"Data loading error: {str(e)}")
    st.stop()


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
            # Ensure data is loaded for the response
            current_data = load_custom_data()
            response = generate_response(prompt, current_data)
        except Exception as e:
            response = f"Error: {str(e)}"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})