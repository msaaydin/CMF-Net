import os
import torch
import random
import numpy as np
from tqdm import tqdm
import copy
import torch.nn as nn
from PIL import Image
import seaborn as sns
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from helpers import read_excel_data
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    return model, model.fc.in_features

# ResNet34 Model Definition
def get_model_resnet34(num_classes):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.in_features

# ResNet50 Model Definition
def get_model_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.in_features

# DenseNet121 Model Definition
def get_model_densenet121(num_classes):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model, model.classifier.in_features

# MobileNetV2 Model Definition
def get_model_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, model.classifier[1].in_features

# MobileNetV3 Model Definition
def get_model_mobilenetv3(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model, model.classifier[3].in_features

def get_model_efficientv2(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, model.classifier[1].in_features


# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.text_features = []
        self.text_data = read_excel_data("data.xlsx")
        
        # Load negative class images
        neg_dir = os.path.join(root_dir, 'negative', split)
        for img_name in os.listdir(neg_dir):
            self.image_paths.append(os.path.join(neg_dir, img_name))
            self.labels.append(0)
            self.text_features.append(self.text_data[split][img_name[:-4]])
            
        # Load positive class images
        pos_dir = os.path.join(root_dir, 'positive', split)
        for img_name in os.listdir(pos_dir):
            self.image_paths.append(os.path.join(pos_dir, img_name))
            self.labels.append(1)
            self.text_features.append(self.text_data[split][img_name[:-4]])
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        #text_feature = self.text_features[]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        text_feature = torch.tensor(self.text_features[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, text_feature
    
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def my_forward(x, model_name, model):
    if 'resnet' in model_name:
        temp_model = nn.Sequential(*list(model.children())[:-1])
        out = F.adaptive_avg_pool2d(temp_model(x), (1, 1))
        embedding = torch.flatten(out, 1)
        del temp_model
        return embedding
    elif 'densenet' in model_name:
        out = F.adaptive_avg_pool2d(model.features(x), (1, 1))
        embedding = torch.flatten(out, 1)
        return embedding
    elif 'mobilenet' in model_name:
        features = model.features(x)
        pooled = F.adaptive_avg_pool2d(features,(1,1))         # → (1, 960, 1, 1)
        embedding = pooled.view(pooled.size(0), -1)  # → (1, 960)
          # → (1, 1280)
        return embedding 
    else:
        temp_model = copy.deepcopy(model)
        temp_model.classifier = nn.Identity()
        return temp_model(x)

class MLPBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPBranch, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.output[1](self.output[0](x))
        x = self.output[3](self.output[2](x))
        return x

class AttentionFusion(nn.Module):
    def __init__(self, cnn_feature_dim, mlp_feature_dim, hidden_dim=128):
        super(AttentionFusion, self).__init__()

        self.cnn_feature_dim = cnn_feature_dim
        self.mlp_feature_dim = mlp_feature_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(cnn_feature_dim + mlp_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 attention weights (CNN and MLP)
            nn.Softmax(dim=1)
        )

        # Project features to the same dimension for fusion
        self.cnn_projection = nn.Linear(cnn_feature_dim, hidden_dim)
        self.mlp_projection = nn.Linear(mlp_feature_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            #nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # Assuming 10 classes
        )

    def forward(self, cnn_features, mlp_features):
        # Project features to same dimension
        cnn_proj = self.cnn_projection(cnn_features)
        mlp_proj = self.mlp_projection(mlp_features)

        # Compute attention weights
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        attention_weights = self.attention(combined)

        # Apply attention weights
        weighted_cnn = cnn_proj * attention_weights[:, 0].unsqueeze(1)
        weighted_mlp = mlp_proj * attention_weights[:, 1].unsqueeze(1)

        # Combine weighted features
        fused_features = weighted_cnn + weighted_mlp

        # Classification
        output = self.classifier(fused_features)

        return output, attention_weights


class MultimodalModel(nn.Module):
    def __init__(self, model_name, mlp_model, attention_model, seed):
        super(MultimodalModel, self).__init__()

        models_dict = {"densenet121": get_model_densenet121, "resnet18": get_model_resnet18,
        "resnet34": get_model_resnet34, "resnet50": get_model_resnet50,
        "mobilenetv2": get_model_mobilenetv2, "mobilenetv3": get_model_mobilenetv3,
        "efficientnet": get_model_efficientv2}
        

        self.model_name = model_name
        self.cnn_branch, self.cnn_featureNum = models_dict[model_name](1)
        self.cnn_branch.load_state_dict(torch.load(f"standart_cnn/model_{model_name}_oneoutput_{seed}.pth", map_location=device))
        self.cnn_branch = self.cnn_branch.to(device)

        self.mlp_branch = mlp_model
        self.mlp_branch.load_state_dict(torch.load("best_model_9091.pth", map_location=device))
        self.mlp_branch = self.mlp_branch.to(device)

        # Attention fusion module
        self.fusion = attention_model

    def forward(self, image, tabular_data):
        # Process image through CNN
        cnn_features = my_forward(image, self.model_name, self.cnn_branch)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        #cnn_features = self.cnn_fc(cnn_features)

        # Process tabular data through MLP
        mlp_features = self.mlp_branch(tabular_data)

        # Fuse features using attention
        output, attention_weights = self.fusion(cnn_features, mlp_features)

        return output

models_dict = {"densenet121": get_model_densenet121, "resnet18": get_model_resnet18,
          "resnet34": get_model_resnet34, "resnet50": get_model_resnet50,
          "mobilenetv2": get_model_mobilenetv2, "mobilenetv3": get_model_mobilenetv3,
          "efficientnet": get_model_efficientv2}


seed_list = {28, 39, 52}
for ii in seed_list:

    cnn_featureNum = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for model_name, model_func in models_dict.items():
        # Model parameters
        input_size = 55
        hidden_size = 128
        output_size = 2  # Number of unique labels
        print(f"Running... {model_name}")
        seed_torch(seed=ii)
        if (model_name == "densenet121"):
            cnn_featureNum = 1024
        elif (model_name == "resnet18"):
            cnn_featureNum = 512
        elif (model_name == "resnet34"):
            cnn_featureNum = 512
        elif (model_name == "resnet50"):
            cnn_featureNum = 2048
        elif (model_name == "mobilenetv2"):
            cnn_featureNum = 1280
        elif (model_name == "mobilenetv3"):
            cnn_featureNum = 960
        elif (model_name == "efficientnet"):
            cnn_featureNum = 1280

        mlp_brach_model = MLPBranch(input_size, hidden_size, output_size)
        attention_mdl = AttentionFusion(cnn_featureNum, 128)
        model = MultimodalModel(model_name, mlp_brach_model, attention_mdl, ii)
        
        model.to(device)

        params = list(model.cnn_branch.parameters()) + \
                list(mlp_brach_model.parameters()) + \
                list(attention_mdl.parameters())


        criterion = nn.BCEWithLogitsLoss()  # Changed loss function
        optimizer = optim.Adam(params, lr=0.001)


        # Training loop with validation
        best_val_loss = float('inf')
        best_val_f1 = float(0)
        num_epochs = 100

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, labels, text_inputs in tqdm(train_loader):
                inputs, labels, text_inputs = inputs.to(device), labels.to(device).float().unsqueeze(1), text_inputs.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs, text_inputs)
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
                for inputs, labels, text_inputs in tqdm(val_loader):
                    inputs, labels, text_inputs = inputs.to(device), labels.to(device).float().unsqueeze(1), text_inputs.to(device)
                
                    outputs = model(inputs, text_inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    probs = torch.sigmoid(outputs)
                    # Threshold at 0.5 for class predictions
                    preds = (probs > 0.5).int()

                    # _, preds = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            val_f1 = f1_score(all_labels, all_preds, average='binary')       
            val_loss = val_loss / len(val_dataset)
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}')
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'cmnet/{model_name}_cmnet_{ii}.pth')
                print('Saved new best model')

        # Testing phase
        model.load_state_dict(torch.load(f'cmnet/{model_name}_cmnet_{ii}.pth'))
        model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels, text_inputs in tqdm(test_loader):
                inputs, labels, text_inputs = inputs.to(device), labels.to(device).float().unsqueeze(1), text_inputs.to(device)
                outputs = model(inputs, text_inputs)
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

        plt.title(f'Confusion Matrix1 acc = {accuracy:.3f}')
        plt.savefig(f'cmnet/model_{model_name}_{ii}.png')
        plt.close()

        result_txt = f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}"

        with open(f"cmnet/{model_name}_result_{ii}.txt", "w") as f:
            f.write(result_txt)

        del mlp_brach_model
        del model
        del attention_mdl


