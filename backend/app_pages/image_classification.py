from backend.pipeline_image_classification import get_unsplash_images, image_classification_loop
import streamlit as st

def write(processor, model, tokeniser):
    st.title('Zero Shot Image Classification - Unsplash Images')
    st.markdown('#')
    st.markdown('#')
    category_image_urls_dict = get_unsplash_images()
    drop_down_choices = [k for k, v in category_image_urls_dict.items()]
    drop_down_choices.sort()
    category = st.selectbox("Choose category", drop_down_choices, index=4)
    image_classification_loop(category, category_image_urls_dict, processor, model, tokeniser)
    return























