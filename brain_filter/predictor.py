#predict.py
import numpy as np
from tensorflow.keras.preprocessing import image
from .setup import model

def is_brain(image_file) -> bool:
    """
    بياخد صورة من request.FILES ويرجع True لو فيها tumor, False otherwise.
    """
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = prediction[0][0] > 0.5

    return bool(result)
