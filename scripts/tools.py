import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_img(img):
    '''
    Plot image
    :param img: image to plot 
    '''
    channels = 1
    if len(np.asarray(img).shape) != 2:
        channels = np.asarray(img).shape[2]
    
    if(channels == 1):
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)

    plt.show()
        
def show_imgs(imgs : list, cols = 5):
    '''
    Plot images
    :param imgs: list of images
    :param cols: number of images in column
    '''
    channels = 1
    if len(np.asarray(imgs[0]).shape) != 2:
        channels = np.asarray(imgs[0]).shape[2]
    
    num_of_images = len(imgs)
    
    if(cols > num_of_images):
        cols = num_of_images
    
    rows = num_of_images / cols #calculate number of rows
    if num_of_images % cols != 0:
        rows += 1
    rows = int(rows)
    
    fig, axs = plt.subplots(rows, cols, figsize = (15, 15), squeeze=False)
    
    index_image = 0
    for row in range(0, rows):
        for col in range(0, cols):
            if index_image >= num_of_images:
                return
            
            if(channels == 1): #show gray image
                axs[row, col].imshow(imgs[index_image], cmap = 'gray')
            else: #show colour image
                axs[row, col].imshow(imgs[index_image])
                
            index_image += 1
                
    plt.show()

def blur(img, debug = False):
    '''
    Blur image
    :param img: image to blur
    :param debug: show result if true
    :return: blurred image
    '''
    res = cv2.GaussianBlur(img, (3, 3), 0)
    
    if debug:
        show_img(res)
        
    return res

def equalize_hist(img, debug = False):
    '''
    Equalize histogram
    :param img: image to equalize
    :param debug: show result if true
    :return: image after equalize histogram
    '''
    equalized_hist = cv2.equalizeHist(img)
    
    if debug:
        show_img(equalized_hist)
        
    return equalized_hist

def colour_equalize_hist(img, debug = False):
    '''
    Equalize histogram of colour image
    :param img: image to equalize
    :param debug: show result if true
    :return: image after equalize histogram
    '''
    equalized_hist = np.copy(img)

    equalized_hist[:,:,0] = equalize_hist(img[:,:,0])
    equalized_hist[:,:,1] = equalize_hist(img[:,:,1])
    equalized_hist[:,:,2] = equalize_hist(img[:,:,2])

    if debug:
        show_img(equalized_hist)

    return equalized_hist

def normalization(img, debug = False):
    '''
    Image normalization
    :param img: image to equalize
    :param debug: show result if true
    :return: image after normalization between (0, 1)
    '''
    normalized = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if debug:
        show_img(normalized)

    return normalized

def preprocess_data(data):
    '''
    preprocess data
    '''
    #data = [colour_equalize_hist(img) for img in data]
    data = [normalization(img) for img in data]
    return data


def preprocess_img(img):
    #img = colour_equalize_hist(img)
    img = normalization(img)
    return img

if __name__ == '__main__':
    img = cv2.imread('sample/0.ppm', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    show_img(img)
    #show_imgs([img[:,:,0], img[:,:,1], img[:,:,2]])

    normalized = normalization(img)
    blurred = blur(img)

    show_imgs([img, normalized, blurred])