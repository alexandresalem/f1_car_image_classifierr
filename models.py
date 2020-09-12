import os
import random

import cv2
import numpy as np
from keras import models
from keras.constraints import maxnorm
from keras.layers import *
from keras.models import Sequential

from constants import IMG_SIZE, TRAIN_FOLDER, F1_CHASSIS_INFO, CONSTRUCTOR_MODEL_PATH
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
    dataset = []
    constructor_list = []
    chassis_dict = {}

    # Mapping folders and converting images into numpy arrays
    for season in os.listdir(folder):
        if int(season) in range(start_year, end_year + 1):
            print(season)
            for car_folder in os.listdir(os.path.join(folder, season)):

                seasons = car_info.get(car_folder)[0]
                car_name = car_info.get(car_folder)[1]
                constructor_name = car_info.get(car_folder)[2]

                if constructor_name not in constructor_list:
                    constructor_list.append(constructor_name)
                    chassis_dict[constructor_name] = [car_folder]

                if car_folder not in chassis_dict.get(constructor_name):
                    chassis_dict[constructor_name].append(car_folder)
                for image in os.listdir(os.path.join(folder, season, car_folder)):
                    try:
                        image_array = cv2.imread(os.path.join(os.environ.get('PWD'), folder, season, car_folder, image))
                        image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))

                        dataset.append((image_array,
                                        car_folder,
                                        constructor_name,
                                        seasons))
                    except:
                        pass
    # Shuffle is a good practice when before making your model train with the dataset
    random.shuffle(dataset)
    return dataset, constructor_list, chassis_dict


def predict_constructor_model(start_year, end_year, building_models):
    """
    Creates a model to predict the constructor's name
    :param building_models: models to be built
    :param start_year: seasons interval - first year
    :param end_year: seasons interval - last year
    :return:
    """

    # Separate dataset into Feature and Target data
    training_dataset, training_constructor_list, training_chassis_dict = images_into_array(start_year, end_year)
    X = []
    y = []
    X.clear()
    y.clear()
    if 'constructor' in building_models:
        for image_array, chassis, constructor, seasons in training_dataset:
            try:
                X.append(image_array)
                y.append(training_constructor_list.index(constructor))
            except:
                pass

        # Transforming data into numpy arrays (to use them in Tensorflow)
        X = np.array(X)
        X = X.astype('float32') / 255
        y = np.array(y)

        constructor_model = create_model(X, y)

        # Saving model into json file
        model_json = constructor_model.to_json()
        with open(f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}.json', 'w') as json_file:
            json_file.write(model_json)
        constructor_model.save_weights(f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}.h5')

        save_json({'results': training_constructor_list},
                  filename=f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}_constructor_results.json')

    if 'chassis' in building_models:
        for training_constructor in training_constructor_list:
            X = []
            y = []
            print(training_constructor)
            for image_array, chassis, constructor, seasons in training_dataset:
                if training_constructor == constructor:
                    try:
                        X.append(image_array)
                        y.append(training_chassis_dict.get(training_constructor).index(chassis))
                    except:
                        pass

            # Transforming data into numpy arrays (to use them in Tensorflow)
            X = np.array(X)
            X = X.astype('float32') / 255
            y = np.array(y)

            model = create_model(X, y)

            # Saving model into json file
            model_json = model.to_json()
            with open(f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}-{training_constructor}.json', 'w') as json_file:
                json_file.write(model_json)
            model.save_weights(f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}-{training_constructor}.h5')

            save_json({'results': training_chassis_dict.get(training_constructor)},
                      filename=f'{CONSTRUCTOR_MODEL_PATH}-{start_year}-{end_year}-{training_constructor}_chassis_results.json')


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
    epochs = 1
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
