import os
import shutil

def reset_data_dir(root_name, alteration_classes):
    directory_to_clear = './' + root_name

    for root, dirs, files in os.walk(directory_to_clear, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)
    
    for pred_class in ['Benign', 'Early', 'Pre', 'Pro']:
        for alt_class in alteration_classes:
            folder_dir = './' + root_name + '/' + pred_class + '/' + alt_class

            if os.path.exists(folder_dir):
                pass
            else:
                os.makedirs(folder_dir)