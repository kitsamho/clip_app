from backend.clip_functions import load_clip
from backend.load import load_yaml
from backend.pipeline_image_classification import get_unsplash_images, image_classification_loop
import streamlit as st



st.set_page_config(layout="wide")

st.title('Zero Shot Image Classification')
st.markdown('#')
st.markdown('#')

config = load_yaml("config.yaml")


category_image_urls_dict = get_unsplash_images()
clip_model = config['CLIP_MODEL'][0]

model, processor, tokeniser = load_clip(clip_model) # load the model, processor and tokeniser - this is cached

drop_down_choices = [k for k, v in category_image_urls_dict.items()]
drop_down_choices.sort()

category = st.sidebar.selectbox("Choose category", drop_down_choices, index=4)

image_classification_loop(category, category_image_urls_dict, processor, model, tokeniser)























