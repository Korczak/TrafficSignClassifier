# TrafficSignClassifier

This repo contains deep learning model, which classify traffic signs (learned on dataset below)
Model is trained to recognize 43 signs.

1. Download dataset from https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip and move data to `data` directory `create directory if necessary`

2. Use predict.py to see predictions

3. To train model use train.py


- image preprocessing: images are normalized, and theirs historgram is equalized.
- load_data uses function to augment data, to create more data to learn model.
- pretrained directory contains trained model.
