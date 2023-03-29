from backend.clip_functions import load_clip
from backend.load import load_yaml, load_cached_embeddings
from backend.pipeline_text_clustering import text_clustering_loop
import streamlit as st

st.set_page_config(layout="wide")

st.title('Text Clustering - Conceptual Captioning Dataset')

config = load_yaml("config.yaml")
clip_model = config['CLIP_MODEL'][0]
cached_embedding_path = 'data/embeddings/text_clustering_embeds_ccaptioning.pickle'
df_cached_embeddings = load_cached_embeddings(cached_embedding_path)
model, processor, tokeniser = load_clip(clip_model) # load the model, processor and tokeniser - this is cached

text_clustering_loop(df_cached_embeddings, tokeniser, model)





