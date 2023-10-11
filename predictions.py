import torch

from capgen.finetune import MyModel

def model_prediction(model_path, image_path):
    model_state_dict = torch.load(model_path)
    # Add a prefix 'model.' to each key in the state dictionary so it can read it
    new_state_dict = {}
    for key, value in model_state_dict.items():
        new_key = 'model.' + key
        new_state_dict[new_key] = value

    mymodel = MyModel() 
    mymodel.load_state_dict(new_state_dict)# Load the state dictionary into the model
    
    return mymodel.gen_caption(image_path)
    
if __name__ == "__main__":
    prediction = model_prediction('./mymodels/Archive/model_2023-10-05 20:37:20_epoch7_wer0.2_4_1e-06_0.0.pth', './Data/Benign/WBC-Benign-001.jpg')
    print(prediction)