from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import streamlit as st


@st.cache_resource
def download_clip_model(clip_model='openai/clip-vit-base-patch32'):
    """
    Load the CLIP model and its associated tokenizer and processor from a given pre-trained model.

    Args:
        clip_model (str): The name or path of the pre-trained model to load.
            Default is 'openai/clip-vit-base-patch32'.

    Returns:
        tuple: A tuple of the CLIP model, tokenizer, and processor.
    """
    model = CLIPModel.from_pretrained(clip_model)
    processor = AutoProcessor.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    return model, processor, tokenizer

def _normalise_features(features):
    """
    Normalize input features by dividing them by their L2 norm.

    Args:
        features (torch.Tensor): Input features to be normalized.

    Returns:
        torch.Tensor: Normalized input features.
    """
    features /= features.norm(dim=-1, keepdim=True)
    return features


def calculate_image_features(images: list, processor, model, normalise=True):
    """
    Calculate image features for a given list of images.

    Args:
        images (list): A list of images to calculate features for.
        processor (transformers.AutoProcessor): The CLIP processor to use.
        model (transformers.CLIPModel): The CLIP model to use.
        normalise (bool): Whether or not to normalize the image features.
            Default is True.

    Returns:
        torch.Tensor: A tensor of the image features for the input images.
    """
    inputs = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    if normalise:
        return _normalise_features(image_features)
    else:
        return image_features


def calculate_text_features(texts: list, tokenizer, model, normalise=True):
    """
    Calculate text features for a given list of texts.

    Args:
        texts (list): A list of texts to calculate features for.
        tokenizer (transformers.AutoTokenizer): The CLIP tokenizer to use.
        model (transformers.CLIPModel): The CLIP model to use.
        normalise (bool): Whether or not to normalize the text features.
            Default is True.

    Returns:
        torch.Tensor: A tensor of the text features for the input texts.
    """
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    if normalise:
        return _normalise_features(text_features)
    else:
        return text_features


def _calculate_similarity(input_features, output_features):
    """
    Calculate the cosine similarity between input and output features.

    Args:
        input_features (torch.Tensor): Input features to compare.
        output_features (torch.Tensor): Output features to compare.

    Returns:
        torch.Tensor: A tensor of the cosine similarity between input and output features.
    """
    similarity = (100.0 * input_features @ output_features.T).softmax(dim=-1)
    return similarity


def classify_images(text_inputs: list, images: list, processor, model, tokeniser):
    """
    Calculates the similarity between the text inputs and images.

    Args:
        text_inputs (list): List of text inputs.
        images (list): List of images.
        processor (transformers.AutoProcessor): The CLIP processor to use.
        model (transformers.CLIPModel): The CLIP model to use.
        tokenizer (transformers.AutoTokenizer): The CLIP tokenizer to use.

    Returns:
        predictions: List of similarity scores between the text inputs and images.
    """
    text_input_features = calculate_text_features(text_inputs, tokeniser, model)
    image_output_features = calculate_image_features(images, processor, model)
    predictions = _calculate_similarity(image_output_features, text_input_features)
    return predictions


def classify_texts(text_inputs: list, texts: list,  model, tokeniser):
    """
    Calculates the similarity between the text inputs and texts.

    Args:
        text_inputs (list): List of text inputs.
        texts (list): List of texts.
        model (transformers.CLIPModel): The CLIP model to use.
        tokenizer (transformers.AutoTokenizer): The CLIP tokenizer to use.

    Returns:
        predictions: List of similarity scores between the text inputs and texts.
    """
    text_input_features = calculate_text_features(text_inputs, tokeniser, model)
    text_output_features = calculate_text_features(texts, tokeniser, model)
    predictions = _calculate_similarity(text_output_features, text_input_features)
    return predictions




