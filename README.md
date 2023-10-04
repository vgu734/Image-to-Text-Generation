# Image-to-Text-Generation

I don't have a list of python packages you may be missing from your environment (sorry, going to be a good number of pip installs before any of these work)
- Some of them may include: torch, torchvision, transformers, jiwer, tqdm, cv2

Make sure you have the data loaded in the root of the directory ./Data/*

train.py will generate altered image from ./Data/* and train a model which will be stored in ./mymodels
- ./mymodels is on the .gitignore because models get big, but don't worry train.py will make that dir if you don't have it
- see ./train_models.sh for an example of the parameters the custom train method takes

validation.py will make a csv of predictions in ./mymodels/Predictions
- again, no need to create ./mymodels/Predictions this file will do it for you
- after you train some models, they will populate in ./mymodels - you will have to specify which model you want to make predictions for 
