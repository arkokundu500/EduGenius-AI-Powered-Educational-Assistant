import streamlit as st
import google.generativeai as genai
import os
from tempfile import NamedTemporaryFile

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_response(prompt, custom_data=None):
    """Generate response using Gemini with optional custom data"""
    if custom_data:
        full_prompt = f"Based on the following information: {custom_data}\n\nAnswer this: {prompt}"
    else:
        full_prompt = prompt
    
    response = model.generate_content(full_prompt)
    return response.text

# Streamlit App
st.set_page_config(page_title="EduChatbot", page_icon="🎓")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "custom_data" not in st.session_state:
    st.session_state.custom_data = None

# Sidebar for custom data input
with st.sidebar:
    st.header("Custom Data Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    if uploaded_file:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            with open(temp_file.name, "r") as f:
                st.session_state.custom_data = f.read()
    
    # Text area input
    text_data = st.text_area("Or paste your text here:", height=200)
    if text_data:
        st.session_state.custom_data = text_data
    
    # Toggle custom data
    use_custom_data = st.toggle("Use Custom Data", value=True)
    
    # Clear data button
    if st.button("Clear Custom Data"):
        st.session_state.custom_data = None

# Main chat interface
st.title("🎓 EduGenuis - Learning Assistant")
st.caption("An AI-powered educational assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    custom_data = st.session_state.custom_data if use_custom_data else None
    with st.spinner("Thinking..."):
        response = generate_response(prompt, custom_data)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})