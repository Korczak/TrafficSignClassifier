import os
import numpy as np

from keras.preprocessing import image
from keras.models import model_from_json

def load_model(name = 'model'):
	json_file = open(name+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights(name+'.h5')

	return model

model = load_model('model0.68_0.88')


images = []
for r, d, f in os.walk('sample'):
	for file in f:
		test_image = image.load_img("sample/{}".format(file), target_size=(32,32))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis=0)
		pred = model.predict(test_image)
		pred = np.argmax(pred)
		print("Prediction for {} is {}".format(file, pred))

