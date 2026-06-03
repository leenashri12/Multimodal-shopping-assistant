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
