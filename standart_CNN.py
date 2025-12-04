import torch
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Load negative class images
        neg_dir = os.path.join(root_dir, 'negative', split)
        for img_name in os.listdir(neg_dir):
            self.image_paths.append(os.path.join(neg_dir, img_name))
            self.labels.append(0)
            
        # Load positive class images
        pos_dir = os.path.join(root_dir, 'positive', split)
        for img_name in os.listdir(pos_dir):
            self.image_paths.append(os.path.join(pos_dir, img_name))
            self.labels.append(1)
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])



# Create datasets and dataloaders
data_dir = 'data'

train_dataset = MyDataset(data_dir, 'train', train_transform)
val_dataset = MyDataset(data_dir, 'val', train_transform)
test_dataset = MyDataset(data_dir, 'test', train_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ResNet18 Model Definition
def get_model_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ResNet34 Model Definition
def get_model_resnet34(num_classes):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ResNet50 Model Definition
def get_model_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# DenseNet121 Model Definition
def get_model_densenet121(num_classes):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# MobileNetV2 Model Definition
def get_model_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# MobileNetV3 Model Definition
def get_model_mobilenetv3(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def get_model_efficientv2(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models_dict = {"densenet121": get_model_densenet121, "resnet18": get_model_resnet18,
          "resnet34": get_model_resnet34, "resnet50": get_model_resnet50,
          "mobilenetv2": get_model_mobilenetv2, "mobilenetv3": get_model_mobilenetv3,
          "efficientnet": get_model_efficientv2}

seed_list = {28, 39, 52}

for ii in seed_list:
    for model_name, model_func in models_dict.items():

        print(f"Running... {model_name}")
        # seed_torch(42)
        seed_torch(seed=ii)
        
        model = model_func(1).to(device)
        criterion = nn.BCEWithLogitsLoss()  # Changed loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop with validation
        best_val_loss = float('inf')
        num_epochs = 100
        best_val_f1 = float(0)

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss = train_loss / len(train_dataset)
            all_labels = []
            all_preds = []
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    probs = torch.sigmoid(outputs)
                    # Threshold at 0.5 for class predictions
                    preds = (probs > 0.5).int()
                    all_preds.extend(preds.cpu().numpy())
                    
            val_loss = val_loss / len(val_dataset)
            val_f1 = f1_score(all_labels, all_preds, average='binary')  
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} validation f1 : {val_f1:.4f}')
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'standart_cnn/model_{model_name}_oneoutput_{ii}.pth')
                print('Saved new best model')

        # Testing phase
        model.load_state_dict(torch.load(f'standart_cnn/model_{model_name}_oneoutput_{ii}.pth'))
        model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                # Threshold at 0.5 for class predictions
                preds = (probs > 0.5).int()

                # _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        print(f'\nTest Results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.title(f'Confusion Matrix {model_name} acc = {accuracy:.3f}')
        plt.savefig(f'standart_cnn/model_{model_name}_{ii}.png')
        plt.close()

        result_txt = f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}"

        with open(f"standart_cnn/{model_name}_result_{ii}.txt", "w") as f:
            f.write(result_txt)
