"""
MED-TRIAGE AI - KAGGLE TRAINING SCRIPT (FULL SCALE)
===================================================

Instructions:
1. Create a New Notebook on Kaggle.
2. Add Data: Search for 'NIH Chest X-ray Dataset' and add it.
3. Copy-paste this entire script into a code cell.
4. Set Accelerator to 'GPU T4 x2'.
5. Run All.
"""

import os
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = '/kaggle/input/nih-chest-x-rays-112635/images_010/images'
CSV_PATH = '/kaggle/input/nih-chest-x-rays-112635/Data_Entry_2017.csv'
BATCH_SIZE = 64
NUM_EPOCHS = 10 
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

print(f"Using Device: {DEVICE}")

# --- DATASET ---
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, labels=LABELS):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.labels = labels
        self.image_paths = self._find_image_paths()
        
    def _find_image_paths(self):
        path_map = {}
        for root, dirs, files in os.walk('/kaggle/input'):
            for file in files:
                if file.endswith('.png'):
                    path_map[file] = os.path.join(root, file)
        return path_map

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        
        image_path = self.image_paths.get(img_name)
        if not image_path:
             image_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        label_vec = torch.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            if label in row['Finding Labels']:
                label_vec[i] = 1.0
                
        return image, label_vec

# --- PREPROCESSING ---
if os.path.exists(CSV_PATH):
    full_df = pd.read_csv(CSV_PATH)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(full_df, test_size=0.1, random_state=42)
    print(f"FULL SCALE TRAINING: {len(train_df)} training images, {len(val_df)} validation images")
else:
    print("WARNING: CSV not found. Please ensure dataset is added.")
    train_df, val_df = pd.DataFrame(), pd.DataFrame()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = ChestXRayDataset(train_df, DATA_DIR, data_transforms['train'])
val_dataset = ChestXRayDataset(val_df, DATA_DIR, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# --- MODEL ---
class ClinicalClassifier(nn.Module):
    def __init__(self, num_classes=len(LABELS)):
        super(ClinicalClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

model = ClinicalClassifier(num_classes=len(LABELS))
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(DEVICE)

# --- TRAINING LOOP ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)

best_model_wts = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
best_loss = float('inf')

print("Starting Full-Scale Training...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'{phase} Loss: {epoch_loss:.4f}')

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_model_wts = copy.deepcopy(state_dict)
            torch.save(state_dict, 'best_model.pth')

time_elapsed = time.time() - start_time
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val loss: {best_loss:.4f}')

torch.save(best_model_wts, 'model_weights.pth')
print("Model saved as 'model_weights.pth'.")
