import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.utils import np_utils

from scripts.tools import *
from scripts.augment_data import augment_data


X_train, y_train, X_test, y_test = [], [], [], []
nb_classes = 43

def load_data():
	'''load data from data folder
	:return: X, y for train and test
	'''
	training_file = "data/train.p"
	test_file = "data/test.p"

	with open(training_file, mode = 'rb') as f:
		train = pickle.load(f)
	with open(test_file, mode = 'rb') as f:
		test = pickle.load(f)

	X_train, y_train = train['features'], train['labels']
	X_test, y_test = test['features'], test['labels']

	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return X_train, y_train, X_test, y_test

def get_number_image_per_class(y_data, debug = False):
	'''
	:param y_data:
	:param debug: show scores as histogram
	:return: array of number of images per class 
	'''
	nb_image_in_class = np.zeros(nb_classes, dtype=np.int64)
	
	nb_hist_classes = []
	for i in range(0, nb_classes):
		nb_hist_classes.append(i)
	for y in y_data:
		for ind, res in enumerate(y):
			if res == 1:
				nb_image_in_class[ind] += 1
	
	if debug:
		plt.bar(nb_hist_classes, nb_image_in_class)
	
	return nb_image_in_class

def get_data_with_class(x_data, y_data, nb_of_class):
	'''get data with specified class
	:param nb_of_class: return data with specified class id
	:return: returns only X_data with specified class id
	'''
	output = []
	
	output.append([x_data[ind] for ind, y in enumerate(y_data) if y[nb_of_class] == 1])
	
	return output[0]

def create_data(x_data, y_data, num_of_data_per_class = 1000):
    '''Creates data with the same number of every classes 
	:param x_data: images from load_data
	:param y_data: labels from load_data
	:return: X_data and y_data - amount = num_of_data_per_class * num_of_classes
    '''
    nb_image_per_class = get_number_image_per_class(y_data)
    data_val = []
    data_imgs = []
    
    
    index_of_x_data = 0
    for ind in range(0, nb_classes):
        data_imgs.append(x_data[index_of_x_data : index_of_x_data + nb_image_per_class[ind]])
        
        index_of_x_data += int(nb_image_per_class[ind])
    
    for ind in range(0, nb_classes):
        while nb_image_per_class[ind] < num_of_data_per_class:
            new_data = np.concatenate((data_imgs[ind], augment_data(data_imgs[ind])))
            data_imgs[ind] = []
            data_imgs[ind] = new_data

            nb_image_per_class[ind] = np.asarray(data_imgs[ind]).shape[0]
            
        data_imgs[ind] = data_imgs[ind][0 : num_of_data_per_class]
        
    data_imgs = np.reshape(data_imgs, (-1, 32, 32, 3))

    data_val = np.zeros((num_of_data_per_class * nb_classes, nb_classes), dtype=np.int8)
    for i in range(0, np.asarray(data_val).shape[0]):
        ind_of_correct_class = int(i / num_of_data_per_class)
        data_val[i][ind_of_correct_class] = 1
    
    
    return data_imgs, data_val


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data();

    create_data(X_train, y_train, num_of_data_per_class = 500)

