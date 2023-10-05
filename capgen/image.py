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
        if 'CONp' in key:
            image = cv2.convertScaleAbs(image, alpha=alt_image_values_dict[key]['CONp'], beta=0)
        if 'CONn' in key:
            image = cv2.convertScaleAbs(image, alpha=alt_image_values_dict[key]['CONn'], beta=0)
        if 'BRIp' in key:
            image = cv2.convertScaleAbs(image, alpha=1, beta=alt_image_values_dict[key]['BRIp'])
        if 'BRIn' in key:
            image = cv2.convertScaleAbs(image, alpha=1, beta=alt_image_values_dict[key]['BRIn'])
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
        if 'ZOOp' in key:
            orig_height, orig_width = image.shape[:2]
            image = cv2.resize(image, 
                               (int(image.shape[0] * alt_image_values_dict[key]['ZOOp']), 
                                int(image.shape[1] * alt_image_values_dict[key]['ZOOp'])), 
                               interpolation=cv2.INTER_LINEAR)
            image = image[0:orig_width, 0:orig_height] #crop or else torch image processor will resize back to 224x224 which undos this
        if 'ZOOn' in key:
            orig_height, orig_width = image.shape[:2]
            image = cv2.resize(image, 
                               (int(image.shape[0] * alt_image_values_dict[key]['ZOOn']), 
                                int(image.shape[1] * alt_image_values_dict[key]['ZOOn'])), 
                               interpolation=cv2.INTER_LINEAR)
            image = image[0:int(.9*image.shape[1]), 0:int(.9*image.shape[0])] #crop out watermark on bottom left corner to avoid reflection replication 
            new_height, new_width = image.shape[:2]
            
            #need to calculate padding for reflections, again we need to do this or torch processor will undo the zoom effect
            top_pad = (orig_height - new_height) // 2
            bottom_pad = orig_height - new_height - top_pad
            left_pad = (orig_width - new_width) // 2
            right_pad = orig_width - new_width - left_pad

            image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_WRAP)

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