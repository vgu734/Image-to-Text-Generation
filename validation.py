import os
import pandas as pd
import random
import torch

from capgen.finetune import MyModel
from capgen.folder import create_gitignore_dirs

from tqdm import tqdm

def model_predictions_for_dir(model_path, path = './Staging', n_files=100):
    model_state_dict = torch.load(model_path)
    # Add a prefix 'model.' to each key in the state dictionary so it can read it
    new_state_dict = {}
    for key, value in model_state_dict.items():
        new_key = 'model.' + key
        new_state_dict[new_key] = value

    mymodel = MyModel() 
    mymodel.load_state_dict(new_state_dict)# Load the state dictionary into the model
    
    predicted_captions = []
    actual_captions = []
    
    total_directories = sum(1 for root, directories, _ in os.walk(path) if root != path and len(root.split(os.path.sep)) - len(path.split(os.path.sep)) == 2)

    progress_bar = tqdm(total=total_directories * n_files, unit="file", desc="Making Predictions")

    predicted_captions = []
    actual_captions = []
    image_paths = []

    for root, directories, files in os.walk(path):
        if root != path:
            parts = root.split(os.path.sep)
            if len(parts) - len(path.split(os.path.sep)) == 2:
                random_files = random.sample(files, min(n_files, len(files)))

                for file in random_files:
                    file_path = os.path.join(root, file)
                    predicted_captions.append(mymodel.gen_caption(file_path))
                    actual_captions.append(file_path.split('/')[4][:file_path.split('/')[4].find('WBC')-1])
                    image_paths.append(file_path)
                    progress_bar.update(1)
    progress_bar.close()
                    
    df = pd.DataFrame({'predicted_captions': predicted_captions, 'actual_captions': actual_captions, 'image_path': image_paths})
    csv_path = model_path.split('/')[-1].replace('.pth', '.csv')
    agg_for_diagnostics(df, csv_path)

def agg_for_diagnostics(df, csv_path):
    df.to_csv(f'./mymodels/Predictions/{csv_path}')
    grouped = df.groupby(["predicted_captions", "actual_captions"]).size().reset_index(name="count")
    pivot_df = grouped.pivot_table(index="actual_captions", columns="predicted_captions", values="count", fill_value=0)
    html_path = csv_path.replace('.csv', '.html')
    pivot_df.to_html(f'./mymodels/Predictions/{html_path}')
    
if __name__ == "__main__":
    create_gitignore_dirs()
    model_predictions_for_dir('./mymodels/Archive/model_2023-10-05 20:37:20_epoch7_wer0.2_4_1e-06_0.0.pth')
