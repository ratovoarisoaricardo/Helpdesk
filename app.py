import os
import sys
import json
import torch
import streamlit as st
import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import EncoderRNN, AttnDecoderRNN
from src.data_utils import prepareData
from src.inference import evaluate

st.set_page_config(page_title="HelpDesk Support", page_icon="🎧", layout="wide")

def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stChatMessage {
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 8px;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.98) !important;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def init_pytorch_model():
    filepath = 'data/faq.json'
    vocab, _ = prepareData(filepath)
    hidden_size = 128
    
    encoder = EncoderRNN(vocab.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, vocab.n_words)
    
    encoder_path = 'models/encoder.pth'
    decoder_path = 'models/decoder.pth'
    
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))
        encoder.eval()
        decoder.eval()
        return encoder, decoder, vocab, True
    return encoder, decoder, vocab, False

def fetch_g4f_response(prompt):
    try:
        import g4f
        sys_prompt = "You are a professional IT HelpDesk support assistant. Provide concise, helpful answers."
        return g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
    except Exception as e:
        return f"Network error (GPT4Free): Please check your internet connection. Details: {e}"

def export_chat_history():
    history = "HelpDesk Support Transcript\n"
    history += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    history += "-" * 40 + "\n\n"
    for msg in st.session_state.messages:
        role = "Support Agent" if msg["role"] == "assistant" else "Customer"
        history += f"{role}: {msg['content']}\n\n"
    return history

def main():
    load_custom_css()
    
    encoder, decoder, vocab, is_trained = init_pytorch_model()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am the IT Support Assistant. How can I help you today?"}
        ]

    # Sidebar configuration
    with st.sidebar:
        st.title("🎧 IT HelpDesk")
        st.markdown("---")
        
        ai_engine = st.radio(
            "Select Engine:", 
            ["🌐 GPT4Free (Smart Cloud AI)", "🧠 PyTorch Seq2Seq (Local FAQ)"]
        )
        
        if "GPT4Free" in ai_engine:
            st.success("Cloud AI Engine is active.")
        else:
            if is_trained:
                st.info("Local PyTorch Model is loaded.")
            else:
                st.error("PyTorch weights not found. Please run train.py")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I assist you?"}]
            st.rerun()
            
        st.markdown("### Export")
        transcript = export_chat_history()
        st.download_button(
            label="📄 Download Transcript",
            data=transcript,
            file_name="support_ticket.txt",
            mime="text/plain"
        )

    # Main chat interface
    st.header("Technical Support Chat")
    st.caption("Powered by advanced AI for rapid problem resolution.")

    for message in st.session_state.messages:
        avatar = "🎧" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Quick suggestion actions
    st.markdown("**Quick Suggestions:**")
    cols = st.columns([1, 1, 1, 1])
    quick_queries = ["Forgot password", "Connection error", "Contact support"]
    
    selected_quick_query = None
    for i, q in enumerate(quick_queries):
        if cols[i].button(q, key=f"quick_{i}"):
            selected_quick_query = q

    user_input = st.chat_input("Type your question here...")
    
    prompt = selected_quick_query if selected_quick_query else user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="🎧"):
            with st.spinner("Analyzing request..."):
                if "GPT4Free" in ai_engine:
                    response = fetch_g4f_response(prompt)
                else:
                    try:
                        response = evaluate(encoder, decoder, vocab, prompt.lower())
                        if not response.strip() or response == "<UNK>":
                            response = "I'm sorry, I couldn't find a solution in my local database."
                    except Exception:
                        response = "I didn't understand those terms."
            
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()