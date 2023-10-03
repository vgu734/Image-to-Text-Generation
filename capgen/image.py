import cv2
import numpy as np
import os
import pandas as pd

from PIL import Image

from capgen.folder import reset_data_dir

def gen_altered_images(alteration_classes, alt_image_values_dict, path='Staging'):
    reset_data_dir(path, alteration_classes)
    
    for key in alt_image_values_dict:
        image = cv2.imread(alt_image_values_dict[key]['orig_path'])
        
        if 'BLUR' in key:
            image = cv2.GaussianBlur(image, (alt_image_values_dict[key]['BLUR'], alt_image_values_dict[key]['BLUR']), 0)
        if 'CONTp' in key:
            image = cv2.convertScaleAbs(image, alpha=alt_image_values_dict[key]['CONTp'], beta=0)
        if 'CONTn' in key:
            image = cv2.convertScaleAbs(image, alpha=alt_image_values_dict[key]['CONTn'], beta=0)
        if 'BRIGHTp' in key:
            image = cv2.convertScaleAbs(image, alpha=1, beta=alt_image_values_dict[key]['BRIGHTp'])
        if 'BRIGHTn' in key:
            image = cv2.convertScaleAbs(image, alpha=1, beta=alt_image_values_dict[key]['BRIGHTn'])
        if 'SATp' in key:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation_factor = alt_image_values_dict[key]['SATp']
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        if 'SATn' in key:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation_factor = alt_image_values_dict[key]['SATn']
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        if 'ZOOMp' in key:
            image = cv2.resize(image, 
                               (int(image.shape[0] * alt_image_values_dict[key]['ZOOMp']), 
                                int(image.shape[1] * alt_image_values_dict[key]['ZOOMp'])), 
                               interpolation=cv2.INTER_LINEAR)
        if 'ZOOMn' in key:
            image = cv2.resize(image, 
                               (int(image.shape[0] * alt_image_values_dict[key]['ZOOMn']), 
                                int(image.shape[1] * alt_image_values_dict[key]['ZOOMn'])), 
                               interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(alt_image_values_dict[key]['staging_path'], image)
        
'''
Return a pandas dataframe containing image path/caption pairs, this is what's fed into the model for fine-tuning
'''
def get_img_text_pairs(alt_image_values_dict):
    image_paths = []
    text_labels = []

    for key in alt_image_values_dict:
        image_paths.append(alt_image_values_dict[key]['staging_path'])
        text_labels.append(alt_image_values_dict[key]['caption'])

    return pd.DataFrame({'image_path': image_paths, 'text': text_labels})