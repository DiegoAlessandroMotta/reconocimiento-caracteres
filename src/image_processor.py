import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> np.ndarray:
    img_np = np.array(image.convert('L'))

    if np.mean(img_np) > 128:
        img_np = 255 - img_np

    img_resized = cv2.resize(img_np, (28, 28), interpolation=cv2.INTER_AREA)

    coords = cv2.findNonZero(img_resized)
    
    digit_img = np.zeros((28, 28), dtype=np.uint8)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        start_x = (28 - w) // 2
        start_y = (28 - h) // 2
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0

        digit_img[start_y:start_y+h, start_x:start_x+w] = img_resized[y:y+h, x:x+w]
    else:
        digit_img = img_resized

    digit_img = digit_img.astype('float32') / 255.0

    return np.expand_dims(digit_img, axis=(0, -1))
