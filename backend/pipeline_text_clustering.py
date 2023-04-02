import streamlit as st
import pandas as pd
import numpy as np
from backend.clip_functions import calculate_text_features
import umap.umap_ as umap
from backend.show import plot_caption_clusters
from backend.dataframes import concatenate_dataframes, get_umap_dataframe


def get_user_input_dataframe(user_input: str, user_embedding: np.array):
    df_user_input = pd.DataFrame({'caption': [user_input], 'category': ['user'], 'embed': [user_embedding]})
    return df_user_input


def delete_row(df, _index):
    return df.iloc[:_index]


def transform_umap_dataframe(df_umap, df_embeddings_merged, user_input):
    df_umap['size'] = 1
    df_umap['text'] = ''
    df_umap.loc[1500, 'size'] = 5
    df_umap.loc[1500, 'text'] = user_input
    df_umap['caption'] = df_embeddings_merged['caption']
    df_umap['category'] = df_embeddings_merged['category']
    return df_umap


def text_clustering_loop(df_text_embeddings, tokeniser, model):
    user_input = st.text_input("Write a short caption related to some of these domains", "birds flying in the sky")
    user_embedding = calculate_text_features(user_input, tokeniser, model, normalise=False)
    df_user_input_embedding = get_user_input_dataframe(user_input, user_embedding.detach().numpy()[0])
    df_embeddings_merged = concatenate_dataframes(df_text_embeddings, df_user_input_embedding)
    df_umap = get_umap_dataframe(umap.UMAP(), df_embeddings_merged.embed.values)
    df_plot = transform_umap_dataframe(df_umap, df_embeddings_merged, user_input)
    fig = plot_caption_clusters(df_plot, x='x', y='y', hover_data=['category', 'caption'], \
                                color='category', size='size', opacity=0.3, text='text')
    st.plotly_chart(fig)
