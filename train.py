import numpy as np
from keras.models import Sequential

from model import get_model, get_model_2
from load_data import load_data, create_data


model = get_model_2()

X_train, y_train, X_test, y_test = load_data()

X_train, y_train = create_data(X_train, y_train)


for epoch in range(0, 5):
	model.fit(x=X_train, y=y_train, batch_size=32, epochs=1,
		verbose=1, callbacks=None, validation_split=0.0, 
		validation_data=None, shuffle=True, class_weight=None, 
		sample_weight=None, initial_epoch=0, steps_per_epoch=None, 
		validation_steps=None)


	loss, metrics = model.evaluate(x=X_test, y=y_test, batch_size=32, verbose=1, sample_weight=None, steps=None)


	model.save_weights("model{:.2f}_{:.2f}.h5".format(loss, metrics))
	model_json = model.to_json()
	with open("model{:.2f}_{:.2f}.json".format(loss, metrics), "w") as json_file:
	    json_file.write(model_json)

	print("Loss: {}, Acc: {}".format(loss, metrics))
