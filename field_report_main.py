import streamlit as st
from PIL import Image
import google.generativeai as genai
from gemini_helper import GeminiInspector

# Page configuration
st.set_page_config(
    page_title="Construction Invoice/Estimate Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'image_analyzed' not in st.session_state:
    st.session_state.image_analyzed = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    if api_key:
        inspector = GeminiInspector(api_key)
    else:
        inspector = GeminiInspector()
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = inspector.start_chat()
        st.session_state.image_analyzed = False
        st.rerun()

# Main title
st.title("📊 Construction Invoice/Estimate Cost Coding Assistant")
st.markdown("Upload an invoice/estimate image to automatically assign cost codes and analyze expenses.")

# Initialize chat if not already done
if st.session_state.chat is None:
    st.session_state.chat = inspector.start_chat()

# File uploader
uploaded_file = st.file_uploader("Upload an invoice/estimate image", type=['png', 'jpg', 'jpeg'])

# Main interface
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Invoice/Estimate Image", use_container_width=True)
    
    # Analyze button
    if not st.session_state.image_analyzed:
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Analyze Invoice/Estimate", type="primary"):
                with st.spinner("Analyzing invoice/estimate and assigning cost codes..."):
                    # Store image for reference
                    st.session_state.current_image = image
                    
                    # Get initial analysis
                    report = inspector.analyze_image(image, st.session_state.chat)
                    
                    # Add the report to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.image_analyzed = True
                    st.rerun()

# Display chat history
st.markdown("### 💬 Invoice/Estimate Analysis Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.image_analyzed:
    if prompt := st.chat_input("Ask questions about the invoice/estimate analysis..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = inspector.send_message(st.session_state.chat, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Footer with instructions
st.markdown("---")
st.markdown("""
### How to Use This Invoice/Estimate Analyzer
1. Upload a clear image of your construction invoice/estimate
2. Click "Analyze Invoice/Estimate" to get an automated cost code analysis
3. Review the generated cost codes and classifications
4. Ask questions about specific line items or classifications
5. Use the "Clear Chat History" button in the sidebar to start fresh

Example questions you can ask:
- Can you explain the reasoning for a specific cost code assignment?
- What items were flagged for review?
- How confident are you about the classifications?
- Can you break down a specific line item into sub-components?
- Are there any alternative cost codes that could apply to this item?
""")

# Download button for chat history
if st.session_state.messages:
    chat_history = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
    st.download_button(
        "Download Analysis Report",
        chat_history,
        file_name="invoice_analysis.txt",
        mime="text/plain"
    )