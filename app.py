import os
import base64
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from transformers import pipeline

# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# ---------------- LOAD SUMMARIZER (CACHED) ----------------
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        token=hf_token,   # ✅ updated
        device=-1         # ✅ CPU safe
    )

summarizer = load_summarizer()

# ---------------- FUNCTION TO SET BACKGROUND IMAGE ----------------
def set_bg_image(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .block-container {{
        background-color: rgba(255,255,255,0.92);
        padding: 20px;
        border-radius: 10px;
    }}

    h1 {{
        color: #0078FF;
        text-align: center;
    }}

    .stButton>button {{
        background-color: #0078FF;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# ✅ FIXED PATH
set_bg_image(r"C:\Users\dighe\OneDrive\Desktop\env\bg.jpg")

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text if text.strip() else None

# ---------------- TEXT CLEAN + LIMIT SIZE ----------------
def clean_text(text, max_chars=3000):
    text = text.replace("\n", " ")
    return text[:max_chars]

# ---------------- STREAMLIT UI ----------------
st.markdown("<h1>Indian Legal Document Summarizer</h1>", unsafe_allow_html=True)
st.write("Upload a legal document PDF to generate a concise summary.")

uploaded_file = st.file_uploader("Drag & Drop or Browse", type=["pdf"])

if uploaded_file:

    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if not text:
        st.error("❌ No readable text found in PDF.")
    else:
        st.success("✅ Text extracted successfully!")

        cleaned_text = clean_text(text)

        with st.spinner("Summarizing document..."):
            summary = summarizer(
                cleaned_text,
                max_length=200,
                min_length=60,
                do_sample=False
            )[0]["summary_text"]

        st.markdown("### 📄 Summary")
        st.info(summary)

        summary_bytes = summary.encode("utf-8")
        b64 = base64.b64encode(summary_bytes).decode()

        download_link = f"""
        <a href="data:file/txt;base64,{b64}" download="summary.txt">
        📥 Download Summary
        </a>
        """

        st.markdown(download_link, unsafe_allow_html=True)

