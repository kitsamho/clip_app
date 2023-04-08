from backend.pipeline_image_classification import get_unsplash_images, image_classification_loop
import streamlit as st
from backend.show import tech_summary_side_bar


def write(processor, model, tokeniser):
    tech_summary_side_bar('zero_shot_classification_imagery')
    st.title('Zero Shot Image Classification - Unsplash Images')
    st.markdown('#')
    st.markdown('#')
    unsplash_urls = get_unsplash_images()
    image_classification_loop(unsplash_urls, processor, model, tokeniser)
    return























