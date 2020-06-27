import numpy as np
import cv2

def random_flip_left_right(image):
    if np.random.randint(2) == 1:
        image = cv2.flip(image, 1)
    return image

def random_brightness(image, alpha=2.0):
    gamma = np.random.rand() * alpha
    gf = [[255 * pow(i/255, 1/gamma)] for i in range(256)]
    table = np.reshape(gf, (256, -1))
    image = cv2.LUT(image, table)
    return image

def random_crop_and_zoom(image, alpha=0.1):
    img_h, img_w = image.shape[:2]
    r = np.random.uniform(0, alpha)
    v1 = np.random.randint(0, int(r*img_h)) if (int(r*img_h) != 0) else 0
    v2 = np.random.randint(0, int(r*img_w)) if (int(r*img_w) != 0) else 0
    image = image[v1:(v1+int((1-r)*img_h)), v2:(v2+int((1-r)*img_w)), :]
    image = cv2.resize(image, (img_h, img_w))
    return image

def normalize(image):
    image = image / 127.5 - 1
    return image

def _augment(image):
    image = cv2.resize(image, (96, 96))
    image = random_flip_left_right(image)
    image = random_crop_and_zoom(image)
    image = random_brightness(image)
    image = normalize(image)
    return image

def augment(images):
    return np.array([_augment(image) for image in images])

