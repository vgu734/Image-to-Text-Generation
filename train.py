import os
import sys
import pandas as pd
import random
import time
import torch

from capgen.data import gen_alt_classes, select_random_images, gen_img_values
from capgen.finetune import MyModel
from capgen.folder import reset_data_dir, create_gitignore_dirs
from capgen.image import gen_altered_images, get_img_text_pairs

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
    
    create_gitignore_dirs()
    
    alteration_classes = gen_alt_classes(base_alt_classes) #maps alteration classes to their captions
    alt_images_dict = select_random_images(alteration_classes) #maps alteration classes to associated random images
    alt_image_values_dict = gen_img_values(alt_images_dict, base_alt_classes) #maps altered images to their paths/randomized alteration parameters/captions
    print("Generating Altered Images...")
    gen_altered_images(alteration_classes, alt_image_values_dict) #generate altered images for each alteration classes and populate Staging
    
    ds = get_img_text_pairs(alt_image_values_dict)
    #print(ds)
    model = MyModel()
    model.train(ds, num_epochs=7, batch_size=int(list({sys.argv[1]})[0]), lr=float(list({sys.argv[2]})[0]), weight_decay=float(list({sys.argv[3]})[0]))
    