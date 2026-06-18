# 6/1/2026
#Goal: takes a pdf, reads it, when asked question give answers from the pdf

import fitz  # PyMuPDF  a powerful python library for manipulation of pdf file

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


import streamlit as st

st.title("PDF AI")

uploaded_file = st.file_uploader("Upload PDF")

if uploaded_file:
    text = extract_text(uploaded_file)
    st.write(text[:1000])

