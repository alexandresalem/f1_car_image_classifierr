import os

import cv2
import numpy as np
from keras.models import model_from_json

from constants import CONSTRUCTOR_MODEL_PATH, IMG_SIZE, TEST_PHOTO_PATH
from utils import load_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_TO_PREDICT = f'{CONSTRUCTOR_MODEL_PATH}-2020-2020'

# load json and create model
with open(f'{MODEL_TO_PREDICT}.json', 'r') as file:
    loaded_model_json = file.read()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"{MODEL_TO_PREDICT}.h5")
print("Loaded model from disk")


try:
    final_image = cv2.imread(TEST_PHOTO_PATH)
    final_image = cv2.resize(final_image, (IMG_SIZE, IMG_SIZE))
except:
    print("Photo wasn't found")

final_image = np.array([final_image])
final_image = final_image.astype(float)/255
prediction = loaded_model.predict(final_image)

result = np.where(prediction[0] == np.amax(prediction[0]))[0][0]
print(prediction)

results = load_json(f'{MODEL_TO_PREDICT}_result_dict.json').get('results', None)

print(f'Foto da equipe {results[result]}')

# for dir in label:
#
#     scuderia = label.index(dir)
#     if scuderia == result[0]:
#         print(f'This car is a {dir}!')
