import os
import sys
import pandas as pd
import random
import time
import torch

from capgen.data import gen_alt_classes, select_random_images, gen_img_values
from capgen.finetune import MyModel
from capgen.folder import reset_data_dir
from capgen.image import gen_altered_images, get_img_text_pairs

def model_predictions_for_dir(alt_image_values_dict, model_path):
    model_state_dict = torch.load(model_path)
    # Add a prefix 'model.' to each key in the state dictionary
    new_state_dict = {}
    for key, value in model_state_dict.items():
        new_key = 'model.' + key
        new_state_dict[new_key] = value

    mymodel = MyModel() 

    # Load the modified state dictionary into the model
    mymodel.load_state_dict(new_state_dict)
    predicted_captions = []
    actual_captions = []
    
    total_items = len(alt_image_values_dict)
    i = 0
    start_time = time.time()
    for key in alt_image_values_dict:
        predicted_captions.append(mymodel.gen_caption(alt_image_values_dict[key]['staging_path']))
        actual_captions.append(alt_image_values_dict[key]['caption'])
        
        progress = (i + 1) / total_items
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # Calculate estimated remaining time
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0

        # Create a simple custom progress bar
        bar_length = 100
        bar = '=' * int(bar_length * progress)
        spaces = ' ' * (bar_length - len(bar))

        # Print the progress bar and ETA
        sys.stdout.write(f'\r[{bar}{spaces}] {int(progress * 100)}% ETA: {int(eta)}s')
        sys.stdout.flush()
        i += 1
        
    df = pd.DataFrame({'predicted_captions': predicted_captions, 'actual_captions': actual_captions})
    csv_path = model_path.split('/')[-1].replace('.pth', '.csv')
    df.to_csv(f'./mymodels/Predictions/{csv_path}')
    return df

if __name__ == "__main__":
    base_alt_classes = { #base alteration classes and their associated captions
    'BLUR': 'increase image resolution', 
    'CONp': 'decrease contrast', 
    'CONn': 'increase contrast', 
    #'BRIp': 'decrease brightness', 
    #'BRIn': 'increase brightness', 
    #'SATp': 'decrease saturation', 
    'SATn': 'increase saturation', 
    #'ZOOp': 'zoom in', 
    #'ZOOn': 'zoom out'
    }
    
    alteration_classes = gen_alt_classes(base_alt_classes) #maps alteration classes to their captions
    alt_images_dict = select_random_images(alteration_classes) #maps alteration classes to associated random images
    alt_image_values_dict = gen_img_values(alt_images_dict, base_alt_classes) #maps altered images to their paths/randomized alteration parameters/captions
    #gen_altered_images(alteration_classes, alt_image_values_dict) #generate altered images for each alteration classes and populate Staging
    
    ds = get_img_text_pairs(alt_image_values_dict)
    #print(ds)
    model = MyModel()
    #model.train(ds[:1000], num_epochs=5)
    model.train(ds, num_epochs=10, batch_size=int(list({sys.argv[1]})[0]), lr=float(list({sys.argv[2]})[0]), weight_decay=float(list({sys.argv[3]})[0]))
    
    #model_predictions_for_dir(alt_image_values_dict, './mymodels/Archive/model_2023-09-30 16:54:45_epoch2_wer0.13.pth')
    