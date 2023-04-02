from backend.pipeline_semantic_search import semantic_search_loop
import streamlit as st


def write(df_cached_embeddings, model, tokenizer):
    st.title('Semantic Search - Rock Archive Images')
    semantic_search_loop(df_cached_embeddings, model, tokenizer)
    return