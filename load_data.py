import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.utils import np_utils

from tools import *


X_train, y_train, X_test, y_test = [], [], [], []
nb_classes = 43

def load_data():

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
	output = []
	
	output.append([x_data[ind] for ind, y in enumerate(y_data) if y[nb_of_class] == 1])
	
	return output[0]

def create_data(x_data, y_data, num_of_data_per_class = 1000):
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
    X_train = preprocess_data(X_train)
    #plt.imshow(X_train[0].squeeze(), cmap='gray')
    #plt.show()
    
    create_data(X_train, y_train)

    #augment_data(X_train, debug = True)