import numpy as np
import pandas as pd
import requests
import umap.umap_ as umap
from PIL import Image
import streamlit as st
from backend.clip_functions import calculate_image_features
from backend.show import plot_image_clusters

from backend.dataframes import concatenate_dataframes, get_umap_dataframe



def get_user_input_dataframe(user_input: str, user_embedding: np.array):
    df_user_input = pd.DataFrame({'artist': ['user'],'image_clip_rep': [user_embedding],\
                                  'url': [user_input]})
    return df_user_input


def image_clustering_loop(df_image_embeddings, processor, model):
    user_image_url = st.text_input("Paste a URL to an image here", "https://media.licdn.com/dms/image/C4D03AQG_rr1doL-1pw/profile-displayphoto-shrink_800_800/0/1570957076434?e=1684972800&v=beta&t=ECN5TDRPRMUK7dma94RkoUQvetMCKyn2vWW2iZyjUDo")
    user_image = Image.open(requests.get(user_image_url, stream=True).raw).convert('L')
    user_image_embedding = calculate_image_features(user_image, processor, model, normalise=False).detach().numpy()[0]
    df_user_image_embedding = get_user_input_dataframe(user_image_url, user_image_embedding)
    df_merged = concatenate_dataframes(df_image_embeddings, df_user_image_embedding)
    df_plot = get_umap_dataframe(umap.UMAP(), df_merged.image_clip_rep.values).join(df_merged)
    df_plot['size'] = [1.5] * df_plot.shape[0]
    df_plot.loc[1250, 'size'] = 30
    plot = plot_image_clusters(df_plot)
    st.altair_chart(plot, use_container_width=True)
    return


