from backend.pipeline_text_clustering import text_clustering_loop
import streamlit as st
from backend.show import tech_summary_side_bar


def write(df_cached_embeddings, tokeniser, model):
    tech_summary_side_bar('clustering')
    st.title('Text Clustering - Conceptual Captioning Dataset')
    text_clustering_loop(df_cached_embeddings, tokeniser, model)
    return





