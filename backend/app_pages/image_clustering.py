from backend.pipeline_image_clustering import image_clustering_loop
import streamlit as st

def write(df_cached_embeddings, processor, model):
    st.title('Image Clustering - Rock Archive Images')
    image_clustering_loop(df_cached_embeddings, processor, model)
    return