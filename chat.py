import streamlit as st
import os
import re
from pathlib import Path
from collections import defaultdict
import io
import time
from pypdf import PdfReader
import docx
from groq import Groq  # Import Groq

CUSTOM_DATA_DIR = "public"  
MAX_CONTEXT_LENGTH = 30000  # Adjust as needed for Llama-3.3's context window
CHUNK_SIZE = 500  

# --- Initialize Groq Client ---
try:
    # Get API key from Streamlit secrets or environment variable
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your environment or Streamlit secrets.")
    
    client = Groq(api_key=groq_api_key)
    LLM_MODEL = "openai/gpt-oss-120b" # The user-specified model

except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()


# Initialize session state for processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- File Extraction Functions (Unchanged) ---

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
        # Create a unique filename to avoid overwrites
        base_name = Path(original_filename).stem
        timestamp = int(time.time())
        new_filename = f"{base_name}_{timestamp}.txt"
        
        save_path = Path(CUSTOM_DATA_DIR) / new_filename
        
        # Ensure public directory exists
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return new_filename
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# --- End of File Extraction Functions ---


@st.cache_data(ttl=600) # Cache data loading for 10 minutes
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

# --- Updated generate_response Function ---
def generate_response(query, custom_data):
    """Enhanced response generation with context prioritization using Groq"""
    relevant_chunks = find_relevant_chunks(query, custom_data)
    
    system_prompt = ""
    context_prompt = ""
    
    if not relevant_chunks:
        system_prompt = "You are an educational assistant. Answer the user's question."
        context_prompt = "(Note: No relevant information found in provided materials. You can upload documents in the sidebar.)"
    else:
        context = "\n\n".join(relevant_chunks)
        # Truncate context if it's too long
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH]
            
        system_prompt = f"""You are an educational assistant. Follow these steps:
1. FIRST Analyze the information within {context} carefully before answering.

    2. Identify key facts or details directly relevant to the question.

    3. If the answer exists in the context, use it exactly as given.

    4. Preserve the original meaning and phrasing from the context.

    5. If the context is incomplete, add information from general educational knowledge.

    6. Clearly state what comes from the context and what comes from general knowledge.

    7. If any conflict appears between the context and general knowledge, follow the context.

    8. Provide clear explanations and logical reasoning for every answer.

    9. Maintain an academic and professional tone in your response.

    10. Never mention or refer to being an AI or language model.

    11. If no answer is found in the context, write:
        “I could not find the information in the given materials, but here are some details from the Web:”

    12. After that line, include only concise and important information from reliable web sources.
"""
        # Context is now in the system prompt, so no extra context prompt is needed
        context_prompt = "" 

    user_message = f"{context_prompt}\n\nQuestion: {query}"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            reasoning_effort="medium",
            model=LLM_MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response from Groq: {e}")
        return "Sorry, I encountered an error while trying to generate a response."

# --- Main App Logic ---

st.set_page_config(page_title="EduGenius - Assistant AI", page_icon="🎓")
st.title("🎓 EduGenius 🎓 your Constant Companion")
st.caption(f"Provide your queries related to educational content")

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

        # Check which files are new
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
                            # Add file ID to processed list
                            st.session_state.processed_files.append(uploaded_file.file_id)
                            files_processed_this_run = True
                        else:
                            st.error(f"Failed to save '{uploaded_file.name}'")
                    else:
                        st.error(f"Could not extract text from '{uploaded_file.name}'")
            
            # Only rerun if new files were processed
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