from backend.clip_functions import load_clip
from backend.load import load_yaml, load_cached_embeddings
from backend.pipeline_semantic_search import semantic_search_loop
import streamlit as st

st.set_page_config(layout="wide")

st.title('Semantic Search - Rock Archive Images')

config = load_yaml("config.yaml")
clip_model = config['CLIP_MODEL'][0]
cached_embedding_path = 'data/embeddings/image_clustering_embeds_rockarchive.pickle'
df_cached_embeddings = load_cached_embeddings(cached_embedding_path)
model, _, tokenizer = load_clip(clip_model) # load the model, processor and tokeniser - this is cached


semantic_search_loop(df_cached_embeddings, model, tokenizer)