import os
import tempfile
import time

import streamlit as st

from shopping_agent import agent, is_shopping_related


def invoke_agent_safely(messages: list) -> str:
    retry_delay = 5
    for attempt in range(3):
        try:
            result = agent.invoke({"messages": messages})
            return result["messages"][-1].content.replace("`", "")
        except Exception as e:
            err_msg = str(e)
            if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                if attempt == 2:  # Last attempt failed, return user-friendly notice
                    return (
                        "⚠️ **Shopping Assistant is Busy (Rate Limit Reached)**: "
                        "The Groq API is experiencing a high volume of requests. "
                        "Please wait 10-15 seconds and try again!"
                    )
                time.sleep(retry_delay)
                retry_delay += 5
            else:
                return f"⚠️ **Error**: {err_msg}"
    return "⚠️ **Error**: Failed to contact the AI assistant."


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")

# Inject Custom CSS for premium design and user experience
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
    }

    /* Gradient Title */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #8b5cf6 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem !important;
    }

    /* Subtitle styling */
    .stCaption {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
    }

    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #111827 0%, #030712 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Chat bubble container design (Glassmorphic) */
    [data-testid="stChatMessage"] {
        background-color: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease !important;
    }

    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    /* User Chat Bubble Accent */
    [data-testid="stChatMessage"][data-avatar="user"] {
        background-color: rgba(139, 92, 246, 0.08) !important;
        border-left: 4px solid #8b5cf6 !important;
    }

    /* Assistant Chat Bubble Accent */
    [data-testid="stChatMessage"][data-avatar="assistant"] {
        background-color: rgba(244, 114, 182, 0.05) !important;
        border-left: 4px solid #f472b6 !important;
    }

    /* Custom Input Bar */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        background-color: #1f2937 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.25) !important;
    }

    /* Gradient buttons with glowing hover effect */
    div.stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #f472b6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px 0 rgba(139, 92, 246, 0.5) !important;
    }

    /* Dashed Upload Frame */
    div[data-testid="stFileUploader"] {
        background-color: rgba(17, 24, 39, 0.6) !important;
        border: 2px dashed rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6 !important;
        background-color: rgba(17, 24, 39, 0.8) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 AI Shopping Assistant")
st.caption("Tell me what you want — I'll search, rate, and order the best match for you.")

# ---------------------------------------------------------------------------
# Sidebar — shop by image
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Shop by Image")
    st.caption("Upload a photo of a product and I'll find similar items in our store.")

    uploaded_file = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if uploaded_file and st.button("Find similar products", use_container_width=True):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        prompt = f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_image = uploaded_file.name
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history — show a friendlier label for image-search messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg["content"].startswith("I uploaded a product image"):
            filename = msg["content"].split("Image path:")[-1].strip()
            st.markdown(f"Searching by image: **{os.path.basename(filename)}**")
        else:
            st.markdown(msg["content"].replace("$", r"\$"))

# ---------------------------------------------------------------------------
# Run agent if there's an unprocessed message (image upload triggers this)
# ---------------------------------------------------------------------------
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and "pending_image" in st.session_state
):
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image and searching…"):
            response = invoke_agent_safely(st.session_state.messages)
        st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({"role": "assistant", "content": response})
    del st.session_state.pending_image
    st.rerun()

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if not is_shopping_related(prompt):
                response = (
                    "I'm sorry, I can only help you with shopping-related queries, "
                    "such as searching for products, checking ratings, viewing your order history, "
                    "or placing orders. Let me know what you'd like to shop for!"
                )
            else:
                response = invoke_agent_safely(st.session_state.messages)
        st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
