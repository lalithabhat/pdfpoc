import streamlit as st
import pandas as pd
import pdfqahelper

col1, col2 = st.columns([3,2])
response_data =''

with col1:
    st.title("Question answering tool")
    query = st.text_area("Paste your  query here", height=300)
    if st.button("Ask"):
        response_data = pdfqahelper.pdf_qa(query)

with col2:
    st.markdown("<br/>" * 5, unsafe_allow_html=True)  # Creates 5 lines of vertical space
    st.write(response_data)