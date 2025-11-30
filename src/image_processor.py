import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    img_np = np.array(image.convert('L'))

    if np.mean(img_np) > 128:
        img_np = 255 - img_np

    coords = cv2.findNonZero(img_np)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        padding = int(max(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_np.shape[1] - x, w + 2 * padding)
        h = min(img_np.shape[0] - y, h + 2 * padding)
        
        digit_cropped = img_np[y:y+h, x:x+w]
    else:
        digit_cropped = img_np

    height, width = digit_cropped.shape
    target_size = 28
    
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    img_resized = cv2.resize(digit_cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    digit_img = np.zeros((target_size, target_size), dtype=np.uint8)
    
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    
    digit_img[start_y:start_y+new_height, start_x:start_x+new_width] = img_resized

    processed_pil = Image.fromarray(digit_img, mode='L')
    
    digit_img_normalized = digit_img.astype('float32') / 255.0

    return digit_img_normalized, processed_pil
