import os
from scipy import misc
import cv2
import random
import numpy as np
from tensorflow.keras.optimizers import *
import tensorflow as tf
from keras.utils import np_utils
from keras.layers import *
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import backend as K
from keras.models import Sequential, model_from_json
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

main_folder = 'f1cars'
label = os.listdir(main_folder)
img_size = 256
dataset = []


#load json and create model
json_file = open('model_f1car.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("model_f1car.h5")
print("Loaded model from disk")




try:
    final_image = cv2.imread('TestLotus.jpg')
    final_image = cv2.resize(final_image, (img_size, img_size))
except:
    print("Photo wasn't found")

final_image = np.array([final_image])
final_image = final_image.astype(float)/255
prediction = loaded_model.predict(final_image)

result = np.where(prediction[0] == np.amax(prediction[0]))
print(prediction)

for dir in label:

    scuderia = label.index(dir)
    if scuderia == result[0]:
        print(f'This car is a {dir}!')
