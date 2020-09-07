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
from constants import F1_CHASSIS, IMG_SIZE, TRAIN_FOLDER, F1_CHASSIS_INFO, CONSTRUCTOR_MODEL_PATH
from utils import load_json, save_json


#
# def generate_list():
#     loaded_json = load_json(F1_CHASSIS)
#     cars = {}
#     for season, chassis in loaded_json.items():
#         for car in chassis:
#             if car not in cars:
#                 renamed_car = re.sub(r"[^a-zA-Z0-9]+", ' ', car)
#                 cars[car] = [[season], renamed_car, car]
#             else:
#                 cars[car][0].append(season)
#
#     save_json(cars, filename='chassis.json')
#
#     chassis_list = []
#     teams_list = []
#     for chassis_season in loaded_json.values():
#         for chassis in chassis_season:
#             chassis = re.sub(r"[^a-zA-Z0-9]+", ' ', chassis)
#             if chassis not in chassis_list:
#                 chassis_list.append(chassis)
#             name = []
#             for word in chassis.split(' '):
#                 if not any(map(str.isdigit, word)):
#                     name.append(word)
#             if (" ".join(name)) not in teams_list:
#                 teams_list.append(" ".join(name))
#     teams_list.sort()
#     if 'teams.txt' not in os.listdir('.'):
#         with open('teams.txt', 'w') as file:
#             for team in teams_list:
#                 file.write(f'{team}\n')
#
#     teams_dict = {}
#     with open('teams.txt', 'r') as file:
#         teams_list.clear()
#         for team in file:
#             teams_list.append(team[:-1])
#
#     for chassis in chassis_list:
#         count = 0
#         for team in teams_list:
#             if team in chassis:
#                 teams_dict[chassis] = team
#                 count += 1
#         if count > 1:
#             print(chassis)
#     return chassis_list, teams_list, teams_dict


def images_into_array(start_year, end_year, folder=TRAIN_FOLDER):
    # FIRST STEP: TRANSFORM IMAGES INTO ARRAYS
    car_info = load_json(os.path.realpath(F1_CHASSIS_INFO))
    print(car_info)
    dataset = []
    results_list = []
    for season in os.listdir(folder):
        if int(season) in range(start_year, end_year + 1):
            print(season)
            for car_folder in os.listdir(os.path.join(folder, season)):
                if car_info.get(car_folder)[2] not in results_list:
                    results_list.append(car_info.get(car_folder)[2])
                for image in os.listdir(os.path.join(folder, season, car_folder)):
                    try:
                        image_array = cv2.imread(os.path.join(os.environ.get('PWD'), folder, season, car_folder, image))
                        image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))

                        dataset.append((image_array,
                                        car_info.get(car_folder)[1],
                                        car_info.get(car_folder)[2],
                                        car_info.get(car_folder)[0]))
                    except:
                        pass
    # Shuffle is a good practice when before making your model train with the dataset
    random.shuffle(dataset)
    return dataset, results_list


def predict_constructor_model(start_year, end_year):
    # Separate dataset into Feature and Target data
    X = []
    y = []
    training_dataset, training_result_list = images_into_array(start_year, end_year)

    for image_array, chassis, team, seasons in training_dataset:
        try:
            X.append(image_array)
            y.append(training_result_list.index(team))
        except:
            pass

    # Transforming data into numpy arrays (to use them in Tensorflow)
    X = np.array(X)
    y = np.array(y)
    X = X.astype('float32') / 255
    model = create_model(X, y)

    # Saving model into json file
    model_json = model.to_json()
    with open(f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}.h5')

    save_json({'results': training_result_list},
              filename=f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}_result_dict.json')

#
# def teams_model():
#     chassis_per_team = {}
#     for team in teams_list:
#         team_cars = []
#         for k, v in teams_dict.items():
#             if v == team and k not in team_cars:
#                 team_cars.append(k)
#         chassis_per_team[team] = team_cars
#     json_dict = json.dumps(chassis_per_team)
#     with open('teams_chassis.json', 'w') as file:
#         file.write(json_dict)
#     # Separate dataset into Feature and Target data
#
#     for team in teams_list:
#         X = []
#         X.clear()
#         y = []
#         y.clear()
#         for image_array, chassis, teams in f1tuple:
#             if team == teams:
#                 try:
#                     X.append(image_array)
#                     y.append(chassis_per_team.get(team).index(chassis))
#
#                 except:
#                     pass
#
#         # Transforming data into numpy arrays (to use them in Tensorflow)
#         if int(len(y)) > 0:
#             print(len(y))
#             X = np.array(X)
#             y = np.array(y)
#             X = X.astype('float32') / 255
#             y = np_utils.to_categorical(y)
#             if y.shape[1] != 1:
#                 model = create_model(X, y)
#                 # Saving model into json file
#                 model_json = model.models.to_json()
#                 with open(f'models/model_f1car_{team}.json', 'w') as json_file:
#                     json_file.write(model_json)
#                 model.save_weights(f'models/model_f1car_{team}.h5')


def create_model(X, y):
    num_classes = len(np.unique(y))
    # Create the model

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same', activation='relu',
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
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #
    # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
    # callbacks=[keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]
    # Fit the model
    model.fit(X, y, epochs=epochs, validation_split=0.1)
    #
    return model
