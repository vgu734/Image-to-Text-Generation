import numpy as np
import torch

from capgen.finetune import MyModel
from PIL import Image

def model_prediction_img_path(model_path, image_path):
    model_state_dict = torch.load(model_path)
    # Add a prefix 'model.' to each key in the state dictionary so it can read it
    new_state_dict = {}
    
    for key, value in model_state_dict.items():
        if key.startswith('model'):
            break
        else:
            for key, value in model_state_dict.items():
                new_key = 'model.' + key
                new_state_dict[new_key] = value
            break

    mymodel = MyModel() 
    mymodel.load_state_dict(new_state_dict)# Load the state dictionary into the model
    
    return mymodel.gen_caption_img_path(image_path)
    
def model_prediction_img_array(model_path, img_array):
    model_state_dict = torch.load(model_path)
    # Add a prefix 'model.' to each key in the state dictionary so it can read it
    new_state_dict = {}
    
    for key, value in model_state_dict.items():
        if key.startswith('model'):
            break
        else:
            for key, value in model_state_dict.items():
                new_key = 'model.' + key
                new_state_dict[new_key] = value
            break

    mymodel = MyModel() 
    mymodel.load_state_dict(new_state_dict)# Load the state dictionary into the model
    
    return mymodel.gen_caption_img_array(img_array)
    
if __name__ == "__main__":
    #make a model prediction given a path to an image
    prediction = model_prediction_img_path('./mymodels/Archive/model_2023-10-05 20:37:20_epoch7_wer0.2_4_1e-06_0.0.pth', './Data/Benign/WBC-Benign-001.jpg')
    print(prediction)
    
    #make a model prediction given an image array
    image = Image.open('./Data/Benign/WBC-Benign-001.jpg')
    img_array = np.array(image)
    prediction = model_prediction_img_array('./mymodels/Archive/model_2023-10-05 20:37:20_epoch7_wer0.2_4_1e-06_0.0.pth', img_array)
    print(prediction)
