import yaml
import streamlit as st
import random


@st.cache_data
def load_yaml(filename):

    with open('backend/'+filename, 'r') as config_file:
        yml_file = yaml.load(config_file, Loader=yaml.FullLoader)
    return yml_file

def get_random_element(_list):
    list_length = len(_list)
    random_index = random.randint(0, list_length)
    return _list[random_index]







