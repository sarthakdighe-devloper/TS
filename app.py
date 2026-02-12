import base64

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Optional: make text readable */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }}
    </style>
    """

    st.markdown(bg_style, unsafe_allow_html=True)
import os
import streamlit as st
import pdfplumber
from transformers import pipeline
from dotenv import load_dotenv

# ---------------- LOAD ENV VARIABLES ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Text Document Summarizer")
st.write("Upload a text or PDF document to generate a summary.")

# ---------------- LOAD SUMMARIZER MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_summarizer():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    return summarizer

summarizer = load_summarizer()

# ---------------- TEXT EXTRACTION FUNCTION ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a TXT or PDF file",
    type=["txt", "pdf"]
)

input_text = ""

if uploaded_file is not None:

    if uploaded_file.type == "application/pdf":
        input_text = extract_text_from_pdf(uploaded_file)

    elif uploaded_file.type == "text/plain":
        input_text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Extracted Text")
    st.text_area("", input_text, height=250)

# ---------------- SUMMARY SETTINGS ----------------
st.sidebar.header("Summary Settings")
max_len = st.sidebar.slider("Max Summary Length", 50, 300, 150)
min_len = st.sidebar.slider("Min Summary Length", 20, 100, 40)

# ---------------- SUMMARIZATION ----------------
if st.button("Generate Summary"):

    if input_text.strip() == "":
        st.warning("Please upload a file first.")
    else:
        with st.spinner("Generating summary..."):

            # Limit input size for model
            input_text = input_text[:3000]

            summary = summarizer(
                input_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )

            st.subheader("✅ Summary")
            st.success(summary[0]["summary_text"])

            # Download option
            st.download_button(
                label="Download Summary",
                data=summary[0]["summary_text"],
                file_name="summary.txt",
                mime="text/plain"
            )

