# 6/1/2026
#Goal: takes a pdf, reads it, when asked question give answers from the pdf
#in python vairable dont hold the data directly it point towards the data in the memory

#streamlit is a tool for making web apps using python
import fitz  # PyMuPDF  a powerful python library for manipulation of pdf file

def extract_text(pdf_file):   #def is the creation of function, funciton name is extract_ text and it takes one input called pdf_file
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf") #stream is the continuous flow of data, it is raw, the data is in uploaded form so the stream takes the data which is read
    text = "" #empty string

    for page in doc: #loop...go through every pages in the loop in the doc
        text += page.get_text()    

    return text

import streamlit as st #imports the streamlit library

st.title("PDF AI")

uploaded_file = st.file_uploader("Upload PDF")

if uploaded_file:
    text = extract_text(uploaded_file)  #If the file is uploaded call the function
    
    st.write(text[:])    


    def chunk_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] #sends chunk of text as the return

    chunks = chunk_text(text)
    st.write(len(chunks))

    # st.write("the end")
    # for chunk in chunks:
    #     st.write(chunk)

from openai import OpenAI
import numpy as np

client = OpenAI(api_key="YOUR_API_KEY") #connects the program to openAI 
                                    #dont do this for real world life application as the key is getting exposed
def get_embedding(text):  #function call
    response = client.embeddings.create(     
        model="text-embedding-3-small",
        input=text
    )     #embedding is the feature under openAI, create is the request sent to openAi server, input=text is the actual data from the pdf sent to the oPENAI server
    return response.data[0].embedding   #the response is a structured object, usually the first one is extracted and sent because it contains the actual embedding

#store chunks
import faiss

index = faiss.IndexFlatL2(1536)
stored_chunks = []

for chunk in chunks:
    emb = get_embedding(chunk)
    index.add(np.array([emb]).astype("float32"))
    stored_chunks.append(chunk)