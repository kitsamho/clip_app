from backend.pipeline_text_clustering import text_clustering_loop
import streamlit as st


def write(df_cached_embeddings, tokeniser, model):
    st.title('Text Clustering - Conceptual Captioning Dataset')
    text_clustering_loop(df_cached_embeddings, tokeniser, model)
    return





