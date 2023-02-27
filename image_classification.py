from backend.pipeline import *
import streamlit as st
st.set_page_config(layout="wide")

st.title('Zero Shot Image Classification')
st.markdown('#')
st.markdown('#')


category_image_urls_dict = get_images()
model, processor, tokeniser = load_model() # load the model, processor and tokeniser - this is cached

drop_down_choices = [k for k, v in category_image_urls_dict.items()]
drop_down_choices.sort()

category = st.sidebar.selectbox("Choose category", drop_down_choices, index=4)

app_loop(category, category_image_urls_dict, processor, model)























