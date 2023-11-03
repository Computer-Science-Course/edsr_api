""" This file is used to test the model on a single image. """
from PIL import Image
import tensorflow as tf

from utils.common_functions import (
    find_model,
    load_edrs_model,
    save_image,
)

model = load_edrs_model(
    find_model('**/*trained.h5')
)

image = Image.open("src/images/john.jpg")
image_tensor = tf.convert_to_tensor(image)

lowres = tf.image.random_crop(image_tensor, (150, 150, 3))
preds = model.predict_step(lowres)

save_image(preds, "src/images/john_super.jpg")
save_image(lowres, "src/images/john_lowres.jpg")
