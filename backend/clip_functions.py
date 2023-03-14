import PIL.Image
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import streamlit as st


@st.cache_resource
def load_clip(clip_model='openai/clip-vit-base-patch32'):
    """
    Load the CLIP model and its associated tokenizer and processor from a given pre-trained model.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model to load.
            Default is 'openai/clip-vit-base-patch32'.

    Returns:
        A tuple of the CLIP model, tokenizer, and processor.
    """
    st.write(f'Loading {clip_model}')
    model = CLIPModel.from_pretrained(clip_model)
    processor = AutoProcessor.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    return model, processor, tokenizer

def _normalise_features(features):
    features /= features.norm(dim=-1, keepdim=True)
    return features


def calculate_image_features(images: list, processor, model):
    """
    Calculate image features for a given list of images.

    Args:
        image (list): A list of images to calculate features for.
        processor: The CLIP processor to use.
        model: The CLIP model to use.

    Returns:
        A numpy array of the image features for the input images.
    """
    inputs = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return _normalise_features(image_features)


def calculate_text_features(texts: list, tokenizer, model):
    """
    Calculate text features for a given list of texts.

    Args:
        text (list): A list of texts to calculate features for.
        tokenizer: The CLIP tokenizer to use.
        model: The CLIP model to use.

    Returns:
        A numpy array of the text features for the input texts.
    """
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return _normalise_features(text_features)


def _calculate_similarity(input_features, output_features):
    similarity = (100.0 * input_features @ output_features.T).softmax(dim=-1)
    return similarity


def classify_images(text_inputs: list, images: list, processor, model, tokeniser):
    text_input_features = calculate_text_features(text_inputs, tokeniser, model)
    image_output_features = calculate_image_features(images, processor, model)
    predictions = _calculate_similarity(image_output_features, text_input_features)
    return predictions


def classify_texts(text_inputs: list, texts: list,  model, tokeniser):
    text_input_features = calculate_text_features(text_inputs, tokeniser, model)
    text_output_features = calculate_text_features(texts, tokeniser, model)
    predictions = _calculate_similarity(text_output_features,text_input_features)
    return predictions



