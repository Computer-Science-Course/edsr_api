import os
from pathlib import Path

from PIL import Image
import numpy as np
import tensorflow as tf

from models.EDSR import EDSRModel


def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(
        high_resolution, super_resolution, max_val=255)[0]
    return psnr_value


def save_image(image_source, filename) -> None:
    """
    Saves unscaled Tensor Images.
    Args:
        image (Tensor): Image tensor
        filename (str): Filename
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    preds_pil = Image.fromarray(np.array(image_source, dtype=np.uint8))
    preds_pil.save(filename)


def find_model(glob_path: str) -> str:
    """Find the model path from the glob path."""
    return list(Path().glob(glob_path))[0]


def load_edrs_model(model_path: str) -> tf.keras.models.Model:
    """Load the EDSR model from the model path."""
    custom_objects = {
        "EDSRModel": EDSRModel,
        'PSNR': PSNR,
    }

    with tf.keras.saving.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(model_path)
