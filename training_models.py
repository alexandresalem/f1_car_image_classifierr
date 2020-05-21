import os
import random

import cv2
import numpy as np
from keras.constraints import maxnorm
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

main_folder = 'download_images/photos'
img_size = 50

def model_all_chassis_all_seasons(img_size):
    dataset = []
    all_teams = []
    print(os.listdir(main_folder))
    for season in os.listdir(main_folder):
        for team in os.listdir(os.path.join(main_folder, season)):
            all_teams.append(team)
            for image in os.listdir(os.path.join(main_folder, season,team)):
                print(os.path.join(main_folder, season, team, image))
                try:
                    img = cv2.imread(os.path.join(os.environ.get('PWD'),main_folder, season, team, image))
                    img = cv2.resize(img, (img_size, img_size))
                    dataset.append((img, team))
                except:
                    pass
                print(len(dataset))


    X = []
    y = []

    random.shuffle(dataset)
    for features, target in dataset:
        X.append(features)
        print(features)
        y.append(all_teams.index(target))

    X = np.array(X)
    y = np.array(y)

    X = X.astype('float32')/255
    y = np_utils.to_categorical(y)

    return X, y


X, y = model_all_chassis_all_seasons(img_size)
num_classes = y.shape[1]

#Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 50
lrate = 0.01
decay = lrate/epochs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#
# callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
# callbacks=[keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]
# Fit the model
model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1)
#
scores = model.evaluate(X,y,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#
# model_json = model.to_json()
# with open('model_f1car.json','w') as json_file:
#     json_file.write(model_json)
# model.save_weights('model_f1car.h5')
