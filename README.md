# Formula 1 - Image Classifier - Deep Learning
Image Classifier - F1 Cars - Using Python and Tensorflow

The code has 3 steps

1) Search the web, for F1 Cars photos, to create trainning data (Webscrapping)

2) Train 'Constructor' and 'Chassis' models. 'Constructor' model predicts the Team name. And each team has a 'Chassis' model to predict the Chassis/Year

(Steps 1 and 2 can be configured in main.py file)

3) Finally, you can run some predictions in predict.py to entertain yourself, or access www.alexandresalem.com/gamef1 to play my version of the game.
  

PS: If your application backend does not support Python, you can use create_jsmodels.py to generate the models in javascript, and use Tensorflowjs to deploy them. 