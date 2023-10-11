import numpy as np
import os
import random

from itertools import combinations

'''
Given a set of base alteration classes, return a dictionary of those classes and combinations of two of those classes as keys that map to a caption for that alteration.
'''
def gen_alt_classes(base_classes):
    alteration_classes = {}
    
    for key in base_classes:
        alteration_classes[key] = base_classes[key]
    for r in range(2, 3):
        for combo in combinations(base_classes, r):
            if combo[0][:3] != combo[1][:3]: #will not map CONp with CONn (doesn't make sense)
                alteration_classes[':'.join(combo)] = base_classes[combo[0]] + ' and ' + base_classes[combo[1]]
    alteration_classes['U'] = ''
    return alteration_classes

'''
Given a set of alteration classes, return a dictionary with alteration class/prediction class keys that map to a list of n_images (100) random images selected for that alteration. The unchanged 'U' class will have 500 random images per prediction class.
'''
def select_random_images(alteration_classes, seed=234234059, n_images=200):
    random.seed(seed)
    
    image_cat_dict = { #There are 504 benign, 985 early, 963 pre and 804 pro images. 
        'Benign': 504,
        'Early': 985,
        'Pre': 963,
        'Pro': 804
    }
    
    alt_images_dict = {} #hold all the images randomly chosen for each class/category

    for key in image_cat_dict:
        outer_key = key
        lower_bound = 1 
        upper_bound = image_cat_dict[outer_key]

        if outer_key == 'Benign':
            img_path_substring = 'WBC-'
        else:
            img_path_substring = 'WBC-Malignant-'

        for key in alteration_classes:
            inner_key = key
            s =  outer_key + '_' + inner_key

            if inner_key == 'U':
                random_numbers = sorted(random.choices(range(lower_bound, upper_bound + 1), k=n_images*5))
            else:
                random_numbers = sorted(random.choices(range(lower_bound, upper_bound + 1), k=n_images))

            file_dir = ['./Data/' + outer_key + '/' + img_path_substring + outer_key + '-{:03d}.jpg'.format(num) 
                        for num in random_numbers]

            alt_images_dict[s] = file_dir
            
    return alt_images_dict
    
'''
Generates a dictionary that maps each altered image to their paths/randomized alteration parameters/captions
'''
def gen_img_values(alt_images_dict, alt_classes, seed=234234059):
    random.seed(seed)
    
    alt_image_values_dict = {}

    for key in alt_images_dict:
        outer_key = key
        inner_args = {}

        if 'BLUR' in key:
            inner_args['BLUR'] = [9, 15 + 1, 2] #correspond to lower_bound, upper_bound, step
        if 'CONp' in key:
            inner_args['CONp'] = [1.35, 1.7, .05]
        if 'CONn' in key:
            inner_args['CONn'] = [.1, .3, .1]
        if 'BRIp' in key:
            inner_args['BRIp'] = [55, 75, 1]
        if 'BRIn' in key:
            inner_args['BRIn'] = [-170, -135, 1]
        if 'SATp' in key:
            inner_args['SATp'] = [1.8, 1.9, .1]
        if 'SATn' in key:
            inner_args['SATn'] = [.1, .2, .1]
        if 'ZOOp' in key:
            inner_args['ZOOp'] = [1.5, 2, .1]
        if 'ZOOn' in key:
            inner_args['ZOOn'] = [.5, .7, .1]
        else:
            pass

        for image_path in alt_images_dict[key]:
            s = str(outer_key) + '_' + image_path[image_path.rfind('/') + 1:image_path.find('.jpg')]

            inner_dict = {}
            inner_dict['orig_path'] = image_path

            if ':' in outer_key:
                category = outer_key[0:outer_key.find('_')]
                alt_class = s[s.find('_') + 1:s.rfind('_')]
                staging_path = './Staging/' + category + '/' + alt_class + '/'+ s + '.jpg'
                caption = '' + alt_classes[alt_class[0:alt_class.find(':')]] + ' and ' + alt_classes[alt_class[alt_class.find(':') + 1:]]
            else:
                category = outer_key[0:outer_key.find('_')]
                alt_class = outer_key[outer_key.rfind('_') + 1:]
                staging_path = './Staging/' + category + '/' + alt_class + '/'+ s + '.jpg'
                if '_U' in outer_key:
                    caption = 'Good blood smear'
                else:
                    caption = '' + alt_classes[alt_class]
                    
            img_name = image_path[image_path.find('WBC'):]
            inner_dict['staging_path'] = staging_path
            inner_dict['ft_path'] = staging_path.replace('Staging', 'Finetuning')
            
            for key in inner_args:
                random_num = round(np.random.choice(np.arange(inner_args[key][0], inner_args[key][1], inner_args[key][2])), 2)
                inner_dict[key] = random_num

            inner_dict['caption'] = caption

            alt_image_values_dict[s] = inner_dict
                
    return alt_image_values_dict
