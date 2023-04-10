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
    """
    Upon initialisation, pre-computed embeddings for c.1500 images from the conceptual captions dataset are loaded. These are already labelled as one of 7 domains: property, sports, food, musicians, vehicles, illustration and nature
    Users are prompted to type a text input related to one of the domains
    The app uses CLIP to generate image embeddings for this user-inputted text
    The Pre-computed conceptual captions are concatenated with this new user embedding
    The UMAP algorithm is used for dimensionality reduction across the concatenated embedding space
    The resulting markers/clusters are displayed as a scatter plot
    Each marker in the scatter plot represents a text string where similar texts are grouped together

    Args:
        df_text_embeddings (pandas.DataFrame): A dataframe of text embeddings
        tokeniser: Tokeniser to use for pre-processing the text data
        model: model : A pre-trained CLIP model.

    Returns:
        None
    """
    # Take user input for a short caption related to some of the domains in the text data
    user_input = st.text_input("Write a short caption related to some of these domains", "birds flying in the sky")

    # Generate text features for the user input
    user_embedding = calculate_text_features(user_input, tokeniser, model, normalise=False)

    # Create a dataframe of the user input and its corresponding text embedding
    df_user_input_embedding = get_user_input_dataframe(user_input, user_embedding.detach().numpy()[0])

    # Concatenate the dataframes of text embeddings and user input embedding
    df_embeddings_merged = concatenate_dataframes(df_text_embeddings, df_user_input_embedding)

    # Apply UMAP on the concatenated dataframe to obtain a UMAP dataframe
    df_umap = get_umap_dataframe(umap.UMAP(), df_embeddings_merged.embed.values)

    # Transform the UMAP dataframe and obtain the dataframe to be used for plotting
    df_plot = transform_umap_dataframe(df_umap, df_embeddings_merged, user_input)

    # Plot the caption clusters using Plotly
    fig = plot_caption_clusters(df_plot, x='x', y='y', hover_data=['category', 'caption'], \
                                color='category', size='size', opacity=0.3, text='text')

    # Show the plot in streamlit
    st.plotly_chart(fig)
    return
