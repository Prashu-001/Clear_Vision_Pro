from PIL import Image
import numpy as np
import tensorflow as tf
import io

def preprocess_image(img: Image.Image, model_type: str) -> np.ndarray:
    if model_type == "WGAN":
        target_size = (128, 128)
    else:
        target_size = (64, 64)

    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)

    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0

    return np.expand_dims(img_array, axis=0)

def postprocess_image(output: np.ndarray) -> bytes:
    if output.ndim == 4:
        output = output[0]

    # Scale back from [-1, 1] to [0, 255]
    output=((output+1.0)*127.5).clip(0,255).astype(np.uint8)
    img = Image.fromarray(output)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()

def to_tensor(img: Image.Image) -> np.ndarray:
    img = img.resize((128, 128))  # Use highest resolution
    arr = np.array(img).astype(np.float32)
    return (arr / 127.5 - 1.0)[np.newaxis, ...]


