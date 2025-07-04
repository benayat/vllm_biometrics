import base64
from io import BytesIO
from PIL import Image


def encode_image_to_base64(image_path) -> str:
    image = Image.open(image_path, mode='r')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')