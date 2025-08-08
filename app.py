import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from typing import Iterable

# --- Setup ---
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")
if not api_key:
    st.warning("OPEN_AI_KEY not found in secrets.env. Add it before running.")
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="Conference Summarizer", page_icon="🧪", layout="centered")
st.title("🧪 Conference Summarizer")

# --- Sidebar controls ---
st.sidebar.header("Model & Params")
model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 100, 2000, 500, 50)
use_stream = st.sidebar.checkbox("Stream response", value=True)
system_prompt = st.sidebar.text_area(
    "System prompt",
    "You are an expert computer vision scientist.",
    height=100
)

# --- Template & inputs ---
default_template = """Use only official sources from the {conference} website, including the accepted papers, awards, news, workshops, orals, and other relevant sections, to research and answer the following for {theme}:

What: A clear, concise explanation of what the topic is, including the core task, method, or technology, written in accessible but technically accurate language.

Why: Why this topic matters or is timely, highlight its technical, commercial, or societal relevance.

Shifts: What has significantly changed from last year, including any notable transitions in methods, architectures, algorithms, or applications. Be specific about "from what → to what."

Future: Where the field is heading, by including open problems, emerging trends, or priorities that indicate future directions.

Papers: List representative {conference} papers that best illustrate the state-of-the-art for this topic, prioritize papers with awards and recognition, and papers with nuances highlighted above.

Limit the output to very short five paragraphs, one per element.
"""

st.subheader("Prompt Template")
prompt_template = st.text_area("Edit your template (use {conference} and {theme})", default_template, height=300)

col1, col2 = st.columns(2)
with col1:
    conference = st.text_input("Conference", "CVPR 2025")
with col2:
    theme = st.text_input("Theme", "3D from multi-view and sensors")

run = st.button("▶️ Run")

# --- Helpers ---
def stream_chat(model: str, messages: list, temperature: float, max_tokens: int) -> Iterable[str]:
    """Yield text chunks from a streaming chat completion."""
    with client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ) as stream:
        for event in stream:
            if event.choices and event.choices[0].delta:
                chunk = event.choices[0].delta.get("content") or ""
                if chunk:
                    yield chunk

# --- Run inference ---
if run:
    if not conference or not theme:
        st.error("Please fill in both **Conference** and **Theme**.")
    else:
        try:
            user_prompt = prompt_template.format(conference=conference, theme=theme)
        except KeyError as e:
            st.error(f"Template placeholder missing: {e}. Make sure you only use {{conference}} and {{theme}}.")
            st.stop()

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

        st.write("**Final prompt preview:**")
        with st.expander("Show / hide prompt"):
            st.code(user_prompt, language="markdown")

        st.subheader("Response")
        if use_stream:
            # Live token stream
            placeholder = st.empty()
            acc = ""
            try:
                for chunk in stream_chat(model, messages, temperature, max_tokens):
                    acc += chunk
                    placeholder.markdown(acc)
            except Exception as e:
                st.error(f"Streaming error: {e}")
        else:
            # One-shot call
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                st.markdown(content)
                with st.expander("Debug: raw response"):
                    st.write(resp.model_dump())
            except Exception as e:
                st.error(f"Request error: {e}")
