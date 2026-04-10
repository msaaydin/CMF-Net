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
from helpers_updated import read_excel_data_all
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

# ===================== MODEL DEFINITIONS =====================
def get_model_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.in_features

def get_model_resnet34(num_classes):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.in_features

def get_model_resnet50(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.in_features

def get_model_densenet121(num_classes):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model, model.classifier.in_features

def get_model_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, model.classifier[1].in_features

def get_model_mobilenetv3(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model, model.classifier[3].in_features

def get_model_efficientv2(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, model.classifier[1].in_features

# ===================== DATASET (LOO version) =====================
class MyDatasetAll(Dataset):
    """Tüm veriyi yükler, split yok. indices ile subset oluşturulur."""
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.text_features = []

        # Tüm tabular verileri oku (split'siz)
        self.text_data = read_excel_data_all("data_old.xlsx")

        # Negative sınıfı
        neg_dir = os.path.join(data_dir, 'negative')
        for img_name in sorted(os.listdir(neg_dir)):
            img_path = os.path.join(neg_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            self.image_paths.append(img_path)
            self.labels.append(0)
            self.text_features.append(self.text_data[img_name[:-4]])

        # Positive sınıfı
        pos_dir = os.path.join(data_dir, 'positive')
        for img_name in sorted(os.listdir(pos_dir)):
            img_path = os.path.join(pos_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            self.image_paths.append(img_path)
            self.labels.append(1)
            self.text_features.append(self.text_data[img_name[:-4]])
        # Tüm listeleri birlikte shuffle et
        combined = list(zip(self.image_paths, self.labels, self.text_features))
        random.seed(42)
        random.shuffle(combined)
        self.image_paths, self.labels, self.text_features = map(list, zip(*combined))
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        text_feature = torch.tensor(self.text_features[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label, text_feature

# ===================== MODULES =====================
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
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        embedding = pooled.view(pooled.size(0), -1)
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
        self.attention = nn.Sequential(
            nn.Linear(cnn_feature_dim + mlp_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        self.cnn_projection = nn.Linear(cnn_feature_dim, hidden_dim)
        self.mlp_projection = nn.Linear(mlp_feature_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, cnn_features, mlp_features):
        cnn_proj = self.cnn_projection(cnn_features)
        mlp_proj = self.mlp_projection(mlp_features)
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        attention_weights = self.attention(combined)
        weighted_cnn = cnn_proj * attention_weights[:, 0].unsqueeze(1)
        weighted_mlp = mlp_proj * attention_weights[:, 1].unsqueeze(1)
        fused_features = weighted_cnn + weighted_mlp
        output = self.classifier(fused_features)
        return output, attention_weights

class MultimodalModel(nn.Module):
    def __init__(self, model_name, mlp_model, attention_model, seed, device):
        super(MultimodalModel, self).__init__()
        models_dict = {
            "densenet121": get_model_densenet121, "resnet18": get_model_resnet18,
            "resnet34": get_model_resnet34, "resnet50": get_model_resnet50,
            "mobilenetv2": get_model_mobilenetv2, "mobilenetv3": get_model_mobilenetv3,
            "efficientnet": get_model_efficientv2
        }
        self.model_name = model_name
        self.cnn_branch, self.cnn_featureNum = models_dict[model_name](1)
        self.cnn_branch.load_state_dict(
            torch.load(f"standart_cnn/model_{model_name}_oneoutput_{seed}.pth", map_location=device)
        )
        self.cnn_branch = self.cnn_branch.to(device)
        self.mlp_branch = mlp_model
        self.mlp_branch.load_state_dict(torch.load("best_model_9091.pth", map_location=device))
        self.mlp_branch = self.mlp_branch.to(device)
        self.fusion = attention_model

    def forward(self, image, tabular_data):
        cnn_features = my_forward(image, self.model_name, self.cnn_branch)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        mlp_features = self.mlp_branch(tabular_data)
        output, attention_weights = self.fusion(cnn_features, mlp_features)
        return output

# ===================== TRANSFORMS =====================
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# ===================== MAIN LOO LOOP =====================
data_dir = 'data/LOO'
os.makedirs('rev1', exist_ok=True)

# Tüm veriyi bir kez yükle
full_dataset = MyDatasetAll(data_dir, train_transform)
N = len(full_dataset)
print(f"Toplam örnek sayısı: {N}")

# models_dict = {
#     "densenet121": get_model_densenet121, "resnet18": get_model_resnet18,
#     "resnet34": get_model_resnet34, "resnet50": get_model_resnet50,
#     "mobilenetv2": get_model_mobilenetv2, "mobilenetv3": get_model_mobilenetv3,
#     "efficientnet": get_model_efficientv2
# }
models_dict = {
    
    "efficientnet": get_model_efficientv2
}

cnn_feature_map = {
    "densenet121": 1024, "resnet18": 512, "resnet34": 512,
    "resnet50": 2048, "mobilenetv2": 1280, "mobilenetv3": 960,
    "efficientnet": 1280
}
seed_val = {
    "densenet121": 39, "resnet18": 28, "resnet34": 52,
    "resnet50": 52, "mobilenetv2": 39, "mobilenetv3": 39,
    "efficientnet": 52
}
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

seed_list = [28, 39, 52]
num_epochs = 40  # LOO'da sabit epoch (validation yok)


for model_name in models_dict.keys():
    seed_v = seed_val[model_name]
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Seed: {seed_v} | LOO ({N} folds)")
    print(f"{'='*60}")

    all_labels = []
    all_preds = []
    all_probs_list = []

    best_fold_loss = float('inf')
    best_fold_state = None

    for fold_idx in range(N):
        seed_torch(seed=seed_v)

        # Train indices: tüm örnekler - test örneği
        train_indices = list(range(N))
        train_indices.remove(fold_idx)
        test_indices = [fold_idx]

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        test_subset = torch.utils.data.Subset(full_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        # Model oluştur
        input_size = 55
        hidden_size = 128
        output_size = 2
        cnn_featureNum = cnn_feature_map[model_name]

        mlp_branch_model = MLPBranch(input_size, hidden_size, output_size)
        attention_mdl = AttentionFusion(cnn_featureNum, 128)
        model = MultimodalModel(model_name, mlp_branch_model, attention_mdl, seed_v, device)
        model.to(device)

        params = list(model.cnn_branch.parameters()) + \
                    list(mlp_branch_model.parameters()) + \
                    list(attention_mdl.parameters())

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params, lr=0.001)

        # ---- Eğitim (sabit epoch, validation yok) ----
        model.train()
        final_train_loss = 0.0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, labels, text_inputs in tqdm(train_loader, desc=f"  Fold {fold_idx+1}/{N} | Epoch {epoch+1}/{num_epochs}", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                text_inputs = text_inputs.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, text_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)

            final_train_loss = epoch_loss / len(train_subset)

        # ---- Test (tek örnek) ----
        model.eval()
        with torch.no_grad():
            for inputs, labels, text_inputs in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                text_inputs = text_inputs.to(device)

                outputs = model(inputs, text_inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                all_labels.extend(labels.cpu().numpy().flatten().tolist())
                all_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_probs_list.extend(probs.cpu().numpy().flatten().tolist())

        # En düşük train loss'a sahip fold'un modelini best olarak sakla
        if final_train_loss < best_fold_loss:
            best_fold_loss = final_train_loss
            best_fold_state = copy.deepcopy(model.state_dict())

        print(f"  Fold {fold_idx+1}/{N} done | Train Loss: {final_train_loss:.4f} | "
                f"True: {all_labels[-1]} | Pred: {all_preds[-1]}")

        del model, mlp_branch_model, attention_mdl
        torch.cuda.empty_cache()

    # ===================== LOO SONUÇLARI =====================
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    print(f'\nLOO Results for {model_name} (seed={seed_v}):')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    # Confusion Matrix kaydet
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'LOO Confusion Matrix - {model_name} (seed={seed_v}) acc={accuracy:.3f}')
    plt.savefig(f'rev1/loo_{model_name}_{seed_v}_cm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Sonuçları txt olarak kaydet
    result_txt = (
        f"Model: {model_name}\n"
        f"Seed: {seed_v}\n"
        f"Evaluation: Leave-One-Out (N={N})\n"
        f"Epochs per fold: {num_epochs}\n"
        f"---\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-Score: {f1:.4f}\n"
    )
    with open(f"rev1/loo_{model_name}_{seed_v}_result.txt", "w") as f:
        f.write(result_txt)

    # Best model kaydet
    if best_fold_state is not None:
        torch.save(best_fold_state, f'rev1/loo_{model_name}_{seed_v}_best.pth')

    print(f"Saved: rev1/loo_{model_name}_{seed_v}_cm.png")
    print(f"Saved: rev1/loo_{model_name}_{seed_v}_result.txt")
    print(f"Saved: rev1/loo_{model_name}_{seed_v}_best.pth")