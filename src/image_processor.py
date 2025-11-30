import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    img_np = np.array(image.convert('L'))

    img_denoised = cv2.medianBlur(img_np, 3)
    
    alpha = 1.0
    beta = 60
    normalized = cv2.convertScaleAbs(img_denoised, alpha=alpha, beta=beta)
    
    gamma = 0.1
    gamma_corrected = np.power(normalized / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(binary_clean)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        padding = int(max(w, h) * 0.12)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary_clean.shape[1] - x, w + 2 * padding)
        h = min(binary_clean.shape[0] - y, h + 2 * padding)
        
        digit_cropped = binary_clean[y:y+h, x:x+w]
    else:
        digit_cropped = binary_clean

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

    _, digit_img = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY)

    processed_pil = Image.fromarray(digit_img, mode='L')
    
    digit_img_normalized = digit_img.astype('float32') / 255.0

    return digit_img_normalized, processed_pil
