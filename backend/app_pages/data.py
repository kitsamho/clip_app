import streamlit as st
def write():
    st.subheader('Data Sources')
    st.markdown("- [BBC Headlines](https://www.bbc.co.uk/news)")
    st.markdown("- [Unsplash](https://unsplash.com/)")
    st.markdown("- [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)")
    st.markdown("- [Rock Archive Images](https://www.rockarchive.com/)")

    st.subheader('Data Caching')
    st.markdown("- Embeddings for rock archive images and conceptual catptions have been pre computed and can be found at"
                 " this path of the project repository: "
                 "[https://github.com/kitsamho/clip_utils_app/tree/main/data/embeddings](https://github.com/kitsamho/clip_utils_app/tree/main/data/embeddings)")

    st.subheader('Project')
    st.markdown("- [Github Repository](https://github.com/kitsamho/clip_utils_app)")

    st.subheader('Other Resources')
    st.markdown("- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)")
    st.markdown("- [CLIP repository](https://github.com/openai/CLIP)")
    st.markdown("- [CLIP website](https://openai.com/research/clip)")
    st.markdown("- [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/)")

    st.subheader('Disclaimer')
    st.markdown("Please note that any data that I have acquired and used are solely for research and demonstration purposes"
                 " only. I want to emphasize that I have not benefited commercially from their use in any way. The "
                 "information and data presented are not intended to be used for any commercial or financial gain.")


