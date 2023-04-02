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


def image_classification_loop(category, category_image_urls_dict, processor, model, tokeniser):

    c1, c2, c3 = st.columns((3, 2, 5))
    if 'image_keep' not in st.session_state:
        image_urls = category_image_urls_dict[category]
        url = get_random_element(image_urls)
        image = Image.open(requests.get(url, stream=True).raw)
        st.session_state['image_keep'] = image
        c1.image(image, width=400)

    else:
        c1.image(st.session_state['image_keep'], width=400)

    text_input_string = st.text_input('Choose some labels for this image - seperate labels with a comma e.g. "dog, cat"', 'dog, cat')
    labels = [i for i in text_input_string.split(",")]

    predictions = classify_images(labels, st.session_state['image_keep'], processor, model, tokeniser)

    df = results_to_dataframe(predictions, labels)
    c3.subheader('Predicted probabilities')
    c3.plotly_chart(plot_results(df, x_label='labels', y_label='probabilities'))

    more_headlines = st.empty()
    next_headline = more_headlines.button('Get new image')
    if next_headline:
        st.session_state.pop('image_keep')
        more_headlines.empty()
        st.experimental_rerun()

    return

