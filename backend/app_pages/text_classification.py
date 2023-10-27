from backend.pipeline_text_classification import get_bbc_headlines, text_classification_loop
import streamlit as st

from backend.show import tech_summary_side_bar


def write(tokeniser, model):
    tech_summary_side_bar('zero_shot_classification_text')
    st.title('Zero Shot Text Classification - BBC Headlines')
    st.markdown('#')
    st.markdown('#')
    bbc_headlines = get_bbc_headlines()
    text_classification_loop(bbc_headlines, tokeniser, model)
    return






















