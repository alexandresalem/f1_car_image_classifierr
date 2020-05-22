import json
import os
import random
import re

import cv2
import numpy as np
from keras.constraints import maxnorm
from keras.layers import *
from keras import models
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
    return chassis_list, teams_list, teams_dict

def images_into_array(img_size,main_folder):

    # FIRST STEP: TRANSFORM IMAGES INTO ARRAYS
    dataset = []
    for season in os.listdir(main_folder):
        print(season)
        for chassis in os.listdir(os.path.join(main_folder, season)):
            if teams_dict.get(chassis.split(' - ')[1]) == None:
                print(f'Adjust team name {chassis.split(" - ")[1]}')
            if chassis.split(' - ')[1] not in chassis_list:
                print(chassis.split(' - ')[1])

            for image in os.listdir(os.path.join(main_folder, season,chassis)):
                try:
                    img = cv2.imread(os.path.join(os.environ.get('PWD'),main_folder, season, chassis, image))
                    img = cv2.resize(img, (img_size, img_size))
                    dataset.append((img, chassis.split(' - ')[1], teams_dict.get(chassis.split(' - ')[1])))
                except:
                    pass
    random.shuffle(dataset)
    return dataset

def full_model():
    # Separate dataset into Feature and Target data
    X = []
    y = []

    for image_array, chassis, team in f1tuple:
        try:
            X.append(image_array)
            y.append(teams_list.index(team))
        except:
            pass

    # Transforming data into numpy arrays (to use them in Tensorflow)
    X = np.array(X)
    y = np.array(y)
    X = X.astype('float32')/255
    y = np_utils.to_categorical(y)
    model = create_model(X,y)

    #Saving model into json file
    model_json = model.models.to_json()
    with open('models/model_f1car.json', 'w') as json_file:
        json_file.write(model_json)
    # model.save_weights('models/model_f1car.h5')

def teams_model():

    chassis_per_team = {}
    for team in teams_list:
        team_cars = []
        for k, v in teams_dict.items():
            if v == team and k not in team_cars:
                team_cars.append(k)
        chassis_per_team[team] = team_cars
    json_dict = json.dumps(chassis_per_team)
    with open('teams_chassis.json', 'w') as file:
        file.write(json_dict)
    # Separate dataset into Feature and Target data

    for team in teams_list:
        X = []
        X.clear()
        y = []
        y.clear()
        for image_array, chassis, teams in f1tuple:
            if team == teams:
                try:
                    X.append(image_array)
                    y.append(chassis_per_team.get(team).index(chassis))

                except:
                    pass

        # Transforming data into numpy arrays (to use them in Tensorflow)
        if int(len(y)) > 0:
            print(len(y))
            X = np.array(X)
            y = np.array(y)
            X = X.astype('float32')/255
            y = np_utils.to_categorical(y)
            if y.shape[1] != 1:
                model = create_model(X,y)
                #Saving model into json file
                model_json = model.models.to_json()
                with open(f'models/model_f1car_{team}.json', 'w') as json_file:
                    json_file.write(model_json)
                model.save_weights(f'models/model_f1car_{team}.h5')


def create_model(X, y):
    num_classes = y.shape[1]
    # Create the model

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
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
    decay = lrate / epochs
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #
    # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
    # callbacks=[keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]
    # Fit the model
    model.fit(X, y, epochs=epochs, validation_split=0.1)
    #
    return model


chassis_list, teams_list, teams_dict = generate_list()
img_size = 50
main_folder = 'download_images/photos'
f1tuple = images_into_array(img_size, main_folder)
full_model()
teams_model()
