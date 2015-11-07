import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_images(imgs, rows = 1):
    num_of_images = np.asarray(imgs).shape[0]
    
    cols = num_of_images / rows
    if num_of_images % rows != 0:
        cols += 1
    cols = int(cols)
    
    
    fig, axs = plt.subplots(rows, cols, figsize = (10, 3), squeeze=False)
    
    for row in range(0, rows):
        for col in range(0, cols):
            axs[row, col].imshow(imgs[row*cols + col])

def augment_data(data, translate = True, scale = True, rotation = True, brightness = True, debug = False):
    '''
    :param data: only X_data
    :param debug: show 1 image before and after augmentation
    :return : augmented data
    '''
    if debug:
        print("Before: ")
        show_images(data[0:2])
    
    if(translate):
        random_translate(data)
    if(scale):
        random_scale(data)
    if(rotation):
        random_rotation(data)
    if(brightness):
        random_brightness(data)
    
    if debug:
        print("After: ")
        show_images(data[0:2])
        
    return data

def random_translate(data, max_translation = 2):
    rows, cols, channels = data[0].shape
    px = max_translation
    dx, dy = np.random.randint(-px, px, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    data = [cv2.warpAffine(img, M, (rows, cols)) for img in data]
    return data

def random_rotation(data, max_rotation = 10):
    rows, cols, channels = np.asarray(data[0]).shape
    rotation = np.random.randint(-max_rotation, max_rotation, 1)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
    data = [cv2.warpAffine(img, M, (rows, cols)) for img in data]
    return data

def random_scale(data, max_scale = 2):
    rows, cols, channels = data[0].shape
    dx, dy = np.random.randint(-max_scale, max_scale, 2)
    
    
    pts_1 = np.float32([[dx, dy],[rows-dx, dy], [dx, cols-dy], [rows - dx, cols - dy]])
    pts_2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
    
    M = cv2.getPerspectiveTransform(pts_1, pts_2)
    
    data = [cv2.warpPerspective(img, M, (rows, cols)) for img in data]
    return data

def random_brightness(data, max_brightness =  25):
    value = np.uint8(np.random.randint(0, max_brightness, 1))
    for img in data:
        max_val = np.max(img)
        lim = max_val - value
        img[img > lim] = max_val
        img[img <= lim] += value
    
    return data


def preprocess_data(data):
    #data = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in data]
    return data