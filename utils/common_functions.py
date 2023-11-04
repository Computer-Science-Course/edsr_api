from pathlib import Path

import tensorflow as tf

from models.EDSR import EDSRModel


def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(
        high_resolution, super_resolution, max_val=255)[0]
    return psnr_value


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
