from backend.pipeline_image_classification import get_unsplash_images, image_classification_loop
import streamlit as st
from backend.load import load_yaml



def write(processor, model, tokeniser):
    config = load_yaml("config.yaml")
    st.write(config)
    expander_info = config['MORE_INFO'][0]['zero_shot_classification']
    st.title('Zero Shot Image Classification - Unsplash Images')
    st.markdown('#')
    st.markdown('#')
    category_image_urls_dict = get_unsplash_images()
    drop_down_choices = [k for k, v in category_image_urls_dict.items()]
    drop_down_choices.sort()
    category = st.selectbox("Choose category", drop_down_choices, index=4)


    with st.expander("Zero shot classification"):
        st.write(expander_info)


    image_classification_loop(category, category_image_urls_dict, processor, model, tokeniser)
    return























