import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from backend.clip_functions import calculate_text_features
from PIL import Image
import requests
from io import BytesIO
import streamlit as st


def rank_vectors(vector_a: np.ndarray, vectors_n: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    """
    Ranks the similarity of a target vector with a list of other vectors, and returns the results in a DataFrame.

    Args:
        vector_a (np.ndarray): The target vector to compare against other vectors.
        vectors_n (np.ndarray): An array of vectors to compare against the target vector.
        labels (pd.Series): A series of labels corresponding to each vector in vectors_n.

    Returns:
        pd.DataFrame: A DataFrame containing the labels and their corresponding similarity scores sorted in descending order.
    """
    # Calculate cosine similarity between the target vector and all other vectors
    similarities = pd.DataFrame(cosine_similarity(vector_a.reshape(1,-1),np.vstack(vectors_n)).T)

    # Combine labels with similarity scores in a DataFrame
    df_results = pd.concat([labels,similarities],axis=1)

    # Sort the DataFrame by similarity scores in descending order
    return df_results.sort_values(by=0,ascending=False).reset_index(drop=True)


def display_image_grid(urls, cols=3, width=300):
    """
    Displays a grid of images loaded from URLs using Streamlit.

    Args:
        urls (list of str): A list of image URLs.
        cols (int): The number of columns in the grid.
        width (int): The width of each image in pixels.
    """
    # Calculate the number of rows in the grid.
    num_rows = int(np.ceil(len(urls) / cols))

    # Create a placeholder for the grid of images.
    image_grid = np.zeros((num_rows * width, cols * width, 3), dtype=np.uint8)

    # Load each image from the URL and insert it into the grid.
    for i, url in enumerate(urls):
        # Load the image from the URL using requests.
        response = requests.get(url)
        # Open the image using PIL.
        image = Image.open(BytesIO(response.content))
        # Resize the image to the desired width.
        image = image.resize((width, width))
        # Convert the PIL Image to a NumPy array.
        image = np.array(image)
        # Insert the image into the grid.
        row = i // cols
        col = i % cols
        image_grid[row * width:(row + 1) * width, col * width:(col + 1) * width, :] = image

    # Display the grid of images in Streamlit.
    st.image(image_grid, channels="RGB")


def semantic_search_loop(df_cached_embeddings, model, tokenizer):
    search = st.text_input("Describe some images you want to see..", 'A band doing a photo shoot outside')
    search_embedding = calculate_text_features([search], tokenizer, model, normalise=False).detach().numpy()[0]

    df_results = rank_vectors(search_embedding, df_cached_embeddings.image_clip_rep, df_cached_embeddings[['url']])

    display_image_grid(df_results.url.values[:10], 5, 300)
