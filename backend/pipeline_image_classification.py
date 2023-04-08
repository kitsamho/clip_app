import requests
from PIL import Image
from backend.clip_functions import classify_images
from backend.dataframes import results_to_dataframe
from backend.load import load_yaml, get_random_element
from backend.scraping import scrape_unsplash_urls
from backend.show import plot_results
import streamlit as st


@st.cache_data
def get_unsplash_images():
    config = load_yaml("config.yaml")  # get the unsplash image categories - this is cached
    unsplash_categories = config['UNSPLASH_CATEGORIES']
    unsplash_categories.sort()
    # get all the unsplash image category:url dictionary - this is cached
    category_image_urls = scrape_unsplash_urls('https://unsplash.com/t/', unsplash_categories)
    return category_image_urls


def image_classification_loop(unsplash_urls: list, processor, model, tokenizer):
    """
    Displays a random image from the provided list of unsplash URLs, prompts the user to enter some contrasting labels
    for the image, and uses the pre-trained model to predict the probabilities of the provided labels.
    Returns the predicted probabilities in a dataframe that get visualised as a bar plot

    Args:
    unsplash_urls: A list of image URLs
    processor: A processor for the CLIP model.
    model: A pre-trained CLIP model.
    tokenizer: A tokenizer for the CLIP model.

    Returns:
    None
    """

    # Split the layout into columns
    c1, c2, c3 = st.columns((3, 2, 5))

    # Display a random image if none is already being displayed
    c1.subheader('Random Image')
    if 'image_keep' not in st.session_state:
        url = get_random_element(unsplash_urls) # get a random image
        image = Image.open(requests.get(url, stream=True).raw) # open the image
        st.session_state['image_keep'] = image # retain the image as a state
        c1.image(image, width=400) # show the image
    else:
        c1.image(st.session_state['image_keep'], width=400)

    # Prompt the user to enter labels for the image
    text_input_string = st.text_input('Choose some contrasting labels for this image - seperate labels with a comma \
                            e.g. "dog, cat" as this is how labels are split', 'dog, cat' )
    labels = [i for i in text_input_string.split(",")]

    # Use the pre-trained model to predict the probabilities of the provided labels
    predictions = classify_images(labels, st.session_state['image_keep'], processor, model, tokenizer)

    # Display the predicted probabilities in a dataframe
    df = results_to_dataframe(predictions, labels)
    c3.subheader('Predicted probabilities')
    c3.plotly_chart(plot_results(df, x_label='labels', y_label='probabilities', color_discrete_sequence='purple'))

    # Allow the user to get a new image
    another_image= st.empty()
    next_headline = another_image.button('Get new image')
    if next_headline:
        st.session_state.pop('image_keep')
        another_image.empty()
        st.experimental_rerun()

    return

