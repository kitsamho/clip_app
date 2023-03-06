from backend.pipeline_text_classification import *
from backend.clip_functions import load_clip
import streamlit as st



st.set_page_config(layout="wide")

st.title('Zero Shot Text Classification')
st.markdown('#')
st.markdown('#')

config = load_yaml("config.yaml")


bbc_headlines_dict = get_bbc_headlines()
clip_model = config['CLIP_MODEL'][0]

model, processor, tokeniser = load_clip(clip_model) # load the model, processor and tokeniser - this is cached

drop_down_choices = [k for k, v in bbc_headlines_dict.items()]
drop_down_choices.sort()

category = st.sidebar.selectbox("Choose category", drop_down_choices, index=4)
#
text_classification_loop(category, bbc_headlines_dict, processor, model)






















