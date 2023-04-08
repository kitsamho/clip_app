from backend.clip_functions import classify_texts
from backend.dataframes import results_to_dataframe
from backend.load import load_yaml, get_random_element
from backend.scraping import scrape_bbc_headlines
from backend.show import plot_results
import streamlit as st


@st.cache_data
def get_bbc_headlines():
    config = load_yaml("config.yaml")  # get the unsplash image categories - this is cached
    bbc_categories = config['BBC_CATEGORIES']
    bbc_categories.sort()
    # get all the unsplash image category:url dictionary - this is cached
    bbc_headlines = scrape_bbc_headlines('https://www.bbc.co.uk/news/', bbc_categories)
    return bbc_headlines


def text_classification_loop(bbc_headlines, tokeniser, model):

    c1, c2, c3 = st.columns((3, 2, 5))
    if 'text_keep' not in st.session_state:
        headline = get_random_element(bbc_headlines)
        st.session_state['text_keep'] = headline
        c1.subheader('Random BBC headline')
        c1.subheader(f'_"{headline}"_')
        st.markdown('#')
        st.markdown('#')

    else:
        c1.subheader('Random BBC headline')
        c1.subheader(f'_"{st.session_state["text_keep"]}"_')
        st.markdown('#')
        st.markdown('#')

    text_input_string = st.text_input('Choose some labels for this text - seperate labels with a comma e.g. "business headline, sports headline"', "business headline, sports headline")
    labels = [i for i in text_input_string.split(",")]

    probs = classify_texts(labels, st.session_state['text_keep'], model, tokeniser)
    df = results_to_dataframe(probs, labels)
    c3.subheader('Predicted probabilities')
    c3.plotly_chart(plot_results(df, x_label='labels', y_label='probabilities'))

    more_headlines = st.empty()
    next_headline = more_headlines.button('Get new headline')
    if next_headline:
        st.session_state.pop('text_keep')
        more_headlines.empty()
        st.experimental_rerun()

    return
