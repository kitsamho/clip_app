from backend.image_scraping import *
from backend.load import *
from backend.show import *
from backend.model import *
from backend.dataframes import *
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import streamlit as st
import random


@st.cache_data
def get_images():
    config = load_yaml("config.yaml")  # get the unsplash image categories - this is cached
    unsplash_categories = config['UNSPLASH_CATEGORIES']
    unsplash_categories.sort()

    # get all the unsplash image category:url dictionary - this is cached
    category_image_urls = scrape_urls(unsplash_categories)
    return category_image_urls


@st.cache_resource
def load_model(model_name_or_path="openai/clip-vit-base-patch32"):
    """
    Load the CLIP model and its associated tokenizer and processor from a given pre-trained model.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model to load.
            Default is 'openai/clip-vit-base-patch32'.

    Returns:
        A tuple of the CLIP model, tokenizer, and processor.
    """
    model = CLIPModel.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, processor, tokenizer


def app_loop(category, category_image_urls_dict, processor, model):



    c1, c2 = st.columns((3, 5))
    if 'image_keep' not in st.session_state:
        image_urls = category_image_urls_dict[category]
        st.write(len(image_urls))
        url = get_random_element(image_urls)
        image = Image.open(requests.get(url, stream=True).raw)
        st.session_state['image_keep'] = image
        c1.image(image, width=300)



    else:

        c1.image(st.session_state['image_keep'], width=300)

    st.subheader("Choose some ways to describe this image")
    lab_1 = st.text_input('1', )
    lab_2 = st.text_input('2', )
    lab_3 = st.text_input('3', )
    inputs = [lab_1, lab_2, lab_3]

    probs = classify_image(inputs, st.session_state['image_keep'], processor, model)
    df = results_to_dataframe(probs, inputs)
    c2.subheader('Predicted probabilities')
    c2.plotly_chart(plot_results(df, x_label='labels', y_label='probabilities'))

    more_images = st.empty()
    next_image = more_images.button('Get new image')
    if next_image:
        st.session_state.pop('image_keep')
        more_images.empty()
        st.experimental_rerun()

    return
