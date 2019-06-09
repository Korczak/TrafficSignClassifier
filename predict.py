import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


from keras.preprocessing import image
from keras.models import model_from_json
from scripts.tools import preprocess_img

def load_model(name = 'model'):
	'''
	Load model
	:param name: name of saved model
	:return: compiled and trained model
	'''
	json_file = open(name+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights(name+'.h5')

	return model

def predict_image(img, model):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, dsize=(32, 32))
	img = preprocess_img(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	pred = np.argmax(pred)
	print("Prediction for {} is {}".format(file, pred))

model = load_model('pretrained/model0.31_0.93')

for r, d, f in os.walk('sample'): #predict 10 images in sample directory
	for file in f:
		img = cv2.imread("sample/{}".format(file), cv2.IMREAD_COLOR) #load image
		predict_image(img, model)

