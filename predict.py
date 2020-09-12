import os

import cv2
import numpy as np
from keras.models import model_from_json

from constants import CONSTRUCTOR_MODEL_PATH, IMG_SIZE, F1_CHASSIS_INFO
from utils import load_json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_TO_PREDICT = f'{CONSTRUCTOR_MODEL_PATH}-2013-2020'
TEST_PHOTO_PATH = 'test_images/Mclaren.jpeg'  # Change it according to your test


def predict(model_path, photo_path):
    # load json and create model
    with open(f'{model_path}.json', 'r') as file:
        loaded_model_json = file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f"{model_path}.h5")
    print("Loaded model from disk")

    try:
        final_image = cv2.imread(photo_path)
        final_image = cv2.resize(final_image, (IMG_SIZE, IMG_SIZE))
    except:
        print("Photo wasn't found")

    final_image = np.array([final_image])
    final_image = final_image.astype(float) / 255
    prediction = loaded_model.predict(final_image)
    ranking = sorted(prediction[0], reverse=True)
    first = np.where(prediction[0] == np.amax(ranking[0]))[0][0]
    second = np.where(prediction[0] == np.amax(ranking[1]))[0][0]
    third = np.where(prediction[0] == np.amax(ranking[2]))[0][0]

    results = load_json(f'{model_path}_constructor_results.json').get('results', None)

    print(f'1st Option: {results[first]} - {round(ranking[0] * 100, 2)}%')
    print(f'2nd Option {results[second]} - {round(ranking[1] * 100, 2)}%')
    print(f'3rd Option {results[third]} - {round(ranking[2] * 100, 2)}%')

    return results[first], final_image


team, image_array = predict(MODEL_TO_PREDICT, TEST_PHOTO_PATH)

with open(f'{MODEL_TO_PREDICT}-{team}.json', 'r') as file:
    loaded_model_json = file.read()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"{MODEL_TO_PREDICT}-{team}.h5")
print("Loaded model from disk")

prediction2 = loaded_model.predict(image_array)

ranking = sorted(prediction2[0], reverse=True)
first = np.where(prediction2[0] == np.amax(ranking[0]))[0][0]

chassis_results = load_json(f'{MODEL_TO_PREDICT}_{team}_chassis_results.json').get('results', None)

car_info = load_json(os.path.realpath(F1_CHASSIS_INFO))

seasons = f"({', '.join(car_info.get(chassis_results[first])[0])})"

print(f'1st Option: {chassis_results[first]} {seasons} - {round(ranking[0] * 100, 2)}%')
