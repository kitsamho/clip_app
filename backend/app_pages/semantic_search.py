from backend.pipeline_semantic_search import semantic_search_loop
import streamlit as st

from backend.show import tech_summary_side_bar


def write(df_cached_embeddings, model, tokenizer):
    tech_summary_side_bar('semantic_search')
    st.title('Semantic Search - Rock Archive Images')
    semantic_search_loop(df_cached_embeddings, model, tokenizer)
    return