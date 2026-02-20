import streamlit as st
from openai import OpenAI
import os

# Page config for better UI
st.set_page_config(page_title="GenAI Chatbot", page_icon="🤖", layout="wide")

# Initialize OpenAI client SAFELY
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("❌ **OPENAI_API_KEY not found!**")
        st.info("""
        **Fix this:**
        1. Get key from https://platform.openai.com/api-keys
        2. **Option A (Recommended):** Add to `.streamlit/secrets.toml`:
           ```
           OPENAI_API_KEY = "sk-your-actual-key-here"
           ```
        3. **Option B:** Terminal: `export OPENAI_API_KEY="sk-your-key"`
        4. Replace and rerun
        """)
        st.stop()  # Gracefully stop Streamlit (NOT sys.exit())
    
    return OpenAI(api_key=api_key)

# Get client
try:
    client = get_openai_client()
except:
    st.error("Failed to initialize OpenAI client. Check your API key.")
    st.stop()

st.title("🤖 Generative AI Chatbot with ML")
st.caption("Powered by GPT-4o-mini + Sentiment Analysis")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for clear chat
with st.sidebar:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything...", key="chat_input"):
    if not prompt.strip():
        st.warning("Please enter a message!")
        st.rerun()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with streaming effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Respond concisely and helpfully."},
                    *st.session_state.messages
                ],
            )
            full_response = response.choices[0].message.content
        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            st.error(f"API Error: {e}")
        
        # Animate response
        for chunk in [full_response[i:] for i in range(0, len(full_response), 30)]:
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.03)  # Typing effect
        message_placeholder.markdown(full_response)
        
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ using Streamlit + OpenAI*")