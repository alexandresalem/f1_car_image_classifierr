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

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

main_folder = 'f1cars'
label = os.listdir(main_folder)
img_size = 256
dataset = []
# for dir in label:
#     images = os.listdir(main_folder + '/' + dir)
#     for image in images:
#         img = cv2.imread(main_folder+'/'+dir+'/'+image)
#         img = cv2.resize(img,(img_size,img_size))
#         dataset.append((img,dir))
#
# X = []
# y = []
#
# random.shuffle(dataset)
# for features, target in dataset:
#     X.append(features)
#     y.append(label.index(target))
#
# X = np.array(X)
# y = np.array(y)
#
# X = X.astype('float32')/255
# y = np_utils.to_categorical(y)
# num_classes = y.shape[1]
#
# data_set = (X,y)
#
#
# #Create the model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
#
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# # Compile model
# epochs = 50
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) # Adam(lr=1e-3)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print(model.summary())
#
# callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
# callbacks=[keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]
# # Fit the model
# model.fit(X, y, epochs=epochs, batch_size=32,shuffle=True,callbacks=callbacks)
#
# scores = model.evaluate(X,y,verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
#
# model_json = model.to_json()
# with open('model_f1car.json','w') as json_file:
#     json_file.write(model_json)
# model.save_weights('model_f1car.h5')









