import streamlit as st
import pandas as pd
import numpy as np
from backend.clip_functions import calculate_text_features
import umap.umap_ as umap
from backend.show import plot_caption_clusters


@st.cache_data
def load_cached_embeddings(path: str):
    df = pd.read_pickle(path)
    return df


def get_user_input_dataframe(user_input: str, user_embedding: np.array):
    df_user_input = pd.DataFrame({'caption': [user_input], 'category': ['user'], 'embed': [user_embedding]})
    return df_user_input


def concatenate_dataframes(df_caption_embeddings, df_user_input_embedding):
    df = pd.concat([df_caption_embeddings, df_user_input_embedding], ignore_index=True)
    return df


def get_umap_dataframe(model, embeddings):
    arrays = [i.reshape(1, -1) for i in embeddings]
    df_umap = pd.DataFrame(model.fit_transform(np.concatenate(arrays, axis=0)))
    df_umap.columns = ['x', 'y']
    return df_umap


def delete_row(df, _index):
    return df.iloc[:_index]


def transform_umap_dataframe(df_umap, df_embeddings_merged):
    df_umap['size'] = 5
    df_umap.loc[3500, 'size'] = 100
    df_umap['caption'] = df_embeddings_merged['caption']
    df_umap['category'] = df_embeddings_merged['category']
    return df_umap


def text_clustering_loop(df_caption_embeddings, tokeniser, model):
    user_input = st.text_input("Write a short caption related to some of these domains")

    user_embedding = calculate_text_features(user_input, tokeniser, model)
    df_user_input_embedding = get_user_input_dataframe(user_input, user_embedding.detach().numpy())

    df_embeddings_merged = concatenate_dataframes(df_caption_embeddings, df_user_input_embedding)

    df_umap = get_umap_dataframe(umap.UMAP(), df_embeddings_merged.embed.values)

    df_plot = transform_umap_dataframe(df_umap, df_embeddings_merged)

    fig = plot_caption_clusters(df_plot, x='x', y='y', hover_data=['category', 'caption'], \
                                color='category', size='size', opacity=0.7)


    st.plotly_chart(fig)
