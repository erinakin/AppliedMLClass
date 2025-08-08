# Transfer Learning 

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input



def preprocess(image, label):
    """
    Applies normalization and model-compatible preprocessing to an image.
    Input:
        image - Tensor of shape (H, W, 3), dtype float32, range [0, 255]
        label - Corresponding label (int)
    Output:
        image - Preprocessed image, shape (299, 299, 3), normalized
        label - Unchanged label
    """
    image = preprocess_input(image)  # Converts to float32 and normalizes per model spec
    return image, label
