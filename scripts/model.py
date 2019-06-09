import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from keras.models import model_from_json

nb_classes = 43


def get_model(summary = False):
	'''
	Creates specified model
	:return: compiled model
	'''
	model = Sequential()

	model.add(Conv2D(32, (3, 3), input_shape= (32,32,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0,2))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0,2))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(nb_classes, activation='sigmoid'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	if summary:
		model.summary();
	return model


def get_model_2(summary = False):
	'''
	Creates specified model
	:return: compiled model
	'''
	model = Sequential()

	model.add(Conv2D(32, (3, 3), input_shape= (32,32,3), activation='relu'))
	model.add(Conv2D(6, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(16, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())

	model.add(Dense(120, activation='relu'))
	model.add(Dropout(0,2))
	model.add(Dense(84, activation='relu'))
	model.add(Dropout(0,2))
	model.add(Dense(42, activation='relu'))
	model.add(Dense(nb_classes, activation='sigmoid'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	if summary:
		model.summary();
	return model


if __name__ == '__main__':
	get_model(summary = True)