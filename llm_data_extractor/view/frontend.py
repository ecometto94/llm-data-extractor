from io import BytesIO

import pypdf
import streamlit as st

from llm_data_extractor.pipeline import run_pipeline


st.title("LLM Data Extractor")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    pdf = pypdf.PdfReader(BytesIO(uploaded_file.getvalue()))
    text = "\n\n".join([page.extract_text(0) for page in pdf.pages])

@st.cache_resource
def persistdata():
    return {}

with st.container():
    fields = persistdata()
    col1, col2 = st.columns(2)
    with col1:
        k = st.text_input('Key')
    with col2:
        v = st.text_input('Value')
    button = st.button('Add')
    if button:
        if k and v:
            fields[k] = v
    st.write(fields)

if st.button("RUN"):
    result = run_pipeline(
        text=text,
        fields_to_extract=fields
    )
    result_dict = result.model_dump()
    st.write(result_dict)