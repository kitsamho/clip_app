from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import streamlit as st



def calculate_image_features(image: list, processor, model):
    """
    Calculate image features for a given list of images.

    Args:
        image (list): A list of images to calculate features for.
        processor: The CLIP processor to use.
        model: The CLIP model to use.

    Returns:
        A numpy array of the image features for the input images.
    """
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features.detach().numpy()


def calculate_text_features(text: list, tokenizer, model):
    """
    Calculate text features for a given list of texts.

    Args:
        text (list): A list of texts to calculate features for.
        tokenizer: The CLIP tokenizer to use.
        model: The CLIP model to use.

    Returns:
        A numpy array of the text features for the input texts.
    """
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features.detach().numpy()


def classify_image(texts: list, images: list, processor, model):
    """
    Classify a list of images using a list of texts as prompts.

    Args:
        texts (list): A list of texts to use as prompts for image classification.
        images (list): A list of images to classify.
        processor: The CLIP processor to use.
        model: The CLIP model to use.

    Returns:
        A numpy array of the classification probabilities for the input images.
    """
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs
