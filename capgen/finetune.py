import datetime
import torch
import torch.nn as nn

from jiwer import wer
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

class MyModel(nn.Module):  # Inherit from torch.nn.Module
    def __init__(self):
        super(MyModel, self).__init__()
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_wer = float('inf')
        self.best_metrics = [None, None, None, None, None]

    def train(self, ds, num_epochs=1, batch_size=4, test_size=.1, lr=1e-5, weight_decay=0.001):
        print(f'Training model with batch size {batch_size}, learning rate {lr:.0e}, weight decay {weight_decay}')
        self.model.to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        train_ds, test_ds = train_test_split(ds, test_size=test_size)
        train_dataset = ProcessDataset(train_ds, self.processor)
        test_dataset = ProcessDataset(test_ds, self.processor)
        
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        validation_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.model.train()
        for epoch in range(num_epochs):
            train_data_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch+1} Training', leave=False)
            
            for idx, batch in enumerate(train_data_iterator):
                input_ids = batch.pop("input_ids").to(self.device)
                pixel_values = batch.pop("pixel_values").to(self.device)
                attention_mask = batch.pop("attention_mask").to(self.device)
    
                outputs = self.model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                attention_mask=attention_mask,
                                labels=input_ids)
    
                loss = outputs.loss
                loss.backward()
    
                optimizer.step()
                optimizer.zero_grad()
                
            self.model.eval()
            total_wer_score = 0.0
            with torch.no_grad():
                train_data_iterator = tqdm(validation_dataloader, desc=f'Epoch {epoch+1} Validation', leave=False)
                
                for batch in train_data_iterator:
                    input_ids = batch.pop("input_ids").to(self.device)
                    pixel_values = batch.pop("pixel_values").to(self.device)
                    attention_mask = batch.pop("attention_mask").to(self.device)
                    labels = input_ids  # Using input_ids as labels
    
                    outputs = self.model(input_ids=input_ids,
                                    pixel_values=pixel_values,
                                    attention_mask=attention_mask,
                                    labels=labels)
    
                    predicted = outputs.logits.argmax(-1)
                    decoded_labels = self.processor.batch_decode(labels.cpu().numpy(), skip_special_tokens=True)
                    decoded_predictions = self.processor.batch_decode(predicted.cpu().numpy(), skip_special_tokens=True)
    
                    wer_score = wer(decoded_labels, decoded_predictions)
                    total_wer_score += wer_score
    
            average_wer_score = total_wer_score / len(validation_dataloader)
            print(f"WER Score after Epoch {epoch+1}: {average_wer_score}")                
            
            if average_wer_score < self.best_wer:
                best_model_state = self.model.state_dict()
                self.best_metrics[0] = epoch + 1
                self.best_metrics[1] = round(average_wer_score, 2)
                self.best_metrics[2] = batch_size
                self.best_metrics[3] = lr
                self.best_metrics[4] = weight_decay
                
                self.best_wer = average_wer_score
                              
        self.save_model(best_model_state)
    
    def gen_caption(self, image_path):
        self.model.to(self.device)
        self.model.eval()
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        pixel_values = inputs.pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption
    
    def save_model(self, best_model_state):
        torch.save(best_model_state, f'./mymodels/model_epoch{self.best_metrics[0]}_wer{self.best_metrics[1]}_{self.best_metrics[2]}_{self.best_metrics[3]:.0e}_{self.best_metrics[4]}.pth')

class ProcessDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        image = row['image_path']
        
        # Preprocess text using the processor
        text_encoding = self.processor(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        
        image = Image.open(image)
        image = self.transform(image)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'pixel_values': image.permute(0, 2, 1)
        }