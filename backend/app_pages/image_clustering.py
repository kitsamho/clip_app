from backend.pipeline_image_clustering import image_clustering_loop
import streamlit as st

from backend.show import tech_summary_side_bar


def write(df_cached_embeddings, processor, model):
    tech_summary_side_bar('clustering')
    st.title('Image Clustering - Rock Archive Images')
    image_clustering_loop(df_cached_embeddings, processor, model)
    return