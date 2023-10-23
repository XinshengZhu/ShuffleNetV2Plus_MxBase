import os
from PIL import Image
import numpy as np

def main():
    img = Image.open("test.JPEG").convert('RGB')
    input_size = 256
    if img.height <= img.width:
        ratio = input_size / img.height
        w_size = int(img.width * ratio)
        img = img.resize((w_size, input_size), Image.BILINEAR)
    else:
        ratio = input_size / img.width
        h_size = int(img.height * ratio)
        img = img.resize((input_size, h_size), Image.BILINEAR)

    img = np.array(img, dtype=np.float32)
    out_width = 224
    out_height = 224
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]

    img = img / 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img[..., 0] = (img[..., 0] - mean[0]) / std[0]
    img[..., 1] = (img[..., 1] - mean[1]) / std[1]
    img[..., 2] = (img[..., 2] - mean[2]) / std[2]
    img = img.transpose(2, 0, 1)

    img.tofile(os.path.join('test.bin'))
    print("success in preprocess")

if __name__ == "__main__":
    main()