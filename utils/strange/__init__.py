""" This file does the image predicting. """
from PIL import Image
import tensorflow as tf

from utils.common_functions import (
    find_model,
    load_edrs_model,
)
from utils.constants import models


def predict(file, model):
    """ Predicts the image. """
    model_glob_path = models.get(model)
    model = load_edrs_model(
        find_model(model_glob_path)
    )

    image = Image.open(file)
    image_tensor = tf.convert_to_tensor(image)

    return model.predict_step(image_tensor)
