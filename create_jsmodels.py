import os

import tensorflowjs as tfjs
from tensorflow.keras.models import model_from_json

for i,model in enumerate(os.listdir('models')):
    if model[-4:] == 'json':
        print(model)
        h5file = model.split('.')[0]+'.h5'
        if model == 'model_f1car.json':
            team = 'all_teams_ever'
        else:
            team = model.split('.')[0].split('_')[-1]
        print(team)


        # Load model into json file
        with open(f'models/{model}', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        # # load weights into new model

        loaded_model.load_weights(f'models/{h5file}')

        print("Loaded model from disk")

        tfjs_target_dir = os.path.join('models', team)
        if tfjs_target_dir not in os.listdir('models/modelsjs'):
            os.mkdir(f'models/modelsjs/{team}')
        tfjs.converters.save_keras_model(loaded_model, tfjs_target_dir)