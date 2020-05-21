import json
import os
import random
import re

import cv2
import numpy as np
from keras.constraints import maxnorm
from keras.layers import *
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.optimizers import *

# Listing folders
path = 'download_images/photos'
json_file = 'download_images/chassis_season.json'

def generate_list():
    with open(json_file, 'r') as file:
        data = file.read()
    loaded_json = json.loads(data)

    chassis_list=[]
    teams_list=[]
    for chassis_season in loaded_json.values():
        for chassis in chassis_season:
            chassis = re.sub(r"[^a-zA-Z0-9]+", ' ', chassis)
            if chassis not in chassis_list:
                chassis_list.append(chassis)
            name = []
            for word in chassis.split(' '):
                if any(map(str.isdigit, word)) == False:
                    name.append(word)
            if (" ".join(name)) not in teams_list:
                teams_list.append(" ".join(name))
    teams_list.sort()
    if 'teams.txt' not in os.listdir('.'):
        with open('teams.txt','w') as file:
            for team in teams_list:
                file.write(f'{team}\n')

    teams_dict = {}
    with open('teams.txt','r') as file:
        teams_list.clear()
        for team in file:
            teams_list.append(team[:-1])

    for chassis in chassis_list:
        count = 0
        for team in teams_list:
            if team in chassis:
                teams_dict[chassis]=team
                count += 1
        if count > 1:
            print(chassis)
    return chassis_list, teams_dict

chassis_list, teams_dict = generate_list()

main_folder = 'download_images/photos'
img_size = 50

def model_all_chassis_all_seasons(img_size):
    dataset = []

    for season in os.listdir(main_folder):
        print(season)
        for chassis in os.listdir(os.path.join(main_folder, season)):
            if chassis.split(' - ')[1] not in chassis_list:
                print(chassis.split(' - ')[1])

            for image in os.listdir(os.path.join(main_folder, season,chassis)):


                try:
                    img = cv2.imread(os.path.join(os.environ.get('PWD'),main_folder, season, chassis, image))
                    img = cv2.resize(img, (img_size, img_size))
                    dataset.append((img, chassis.split(' - ')[1], teams_dict.get(chassis.split(' - ')[1])))
                except:
                    pass



    X = []
    y = []
    teams_list=[]
    with open('teams.txt', 'r') as file:
        teams_list.clear()
        for team in file:
            teams_list.append(team[:-1])


    random.shuffle(dataset)
    for features, chassis, target in dataset:
        try:
            X.append(features)
            y.append(teams_list.index(target))
        except:
            pass

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
model_json = model.to_json()
with open('model_f1car.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('model_f1car.h5')
