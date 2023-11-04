""" This file does the image predicting. """
from PIL import Image
import tensorflow as tf

from utils.common_functions import (
    find_model,
    load_edrs_model,
)


def predict(file):
    """ Predicts the image. """
    model = load_edrs_model(
        find_model('**/*trained.h5')
    )

    image = Image.open(file)
    image_tensor = tf.convert_to_tensor(image)

    return model.predict_step(image_tensor)
