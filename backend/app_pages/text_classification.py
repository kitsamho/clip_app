from backend.pipeline_text_classification import get_bbc_headlines, text_classification_loop
import streamlit as st


def write(tokeniser, model):
    st.title('Zero Shot Text Classification - BBC Headlines')
    st.markdown('#')
    st.markdown('#')
    bbc_headlines_dict = get_bbc_headlines()
    drop_down_choices = [k for k, v in bbc_headlines_dict.items()]
    drop_down_choices.sort()
    category = st.selectbox("Choose category", drop_down_choices, index=4)
    text_classification_loop(category, bbc_headlines_dict, tokeniser, model)
    return






















