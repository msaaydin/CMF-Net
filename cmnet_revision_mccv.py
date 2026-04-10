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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

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

# ===================== DATASET =====================
class MyDatasetAll(Dataset):
    def __init__(self, data_dir, transform=None, seed=42):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.text_features = []

        self.text_data = read_excel_data_all("data_old.xlsx")

        neg_dir = os.path.join(data_dir, 'negative')
        for img_name in sorted(os.listdir(neg_dir)):
            img_path = os.path.join(neg_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            self.image_paths.append(img_path)
            self.labels.append(0)
            self.text_features.append(self.text_data[img_name[:-4]])

        pos_dir = os.path.join(data_dir, 'positive')
        for img_name in sorted(os.listdir(pos_dir)):
            img_path = os.path.join(pos_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            self.image_paths.append(img_path)
            self.labels.append(1)
            self.text_features.append(self.text_data[img_name[:-4]])

        # Shuffle all lists together
        combined = list(zip(self.image_paths, self.labels, self.text_features))
        random.seed(seed)
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

# ===================== CONFIG =====================
data_dir = 'data/LOO'
os.makedirs('rev_mccv', exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Her model için en iyi seed
seed_val = {
    "densenet121": 39, "resnet18": 28, "resnet34": 52,
    "resnet50": 52, "mobilenetv2": 39, "mobilenetv3": 39,
    "efficientnet": 52
}

models_dict = {
    "densenet121": get_model_densenet121, "resnet18": get_model_resnet18,
    "resnet34": get_model_resnet34, "resnet50": get_model_resnet50,
    "mobilenetv2": get_model_mobilenetv2, "mobilenetv3": get_model_mobilenetv3,
    "efficientnet": get_model_efficientv2
}

cnn_feature_map = {
    "densenet121": 1024, "resnet18": 512, "resnet34": 512,
    "resnet50": 2048, "mobilenetv2": 1280, "mobilenetv3": 960,
    "efficientnet": 1280
}

K = 100           # MCCV iterasyon sayısı
num_epochs = 100  # Validation ile early stopping olduğu için daha yüksek tutabiliriz
patience = 10     # Early stopping patience

# Tüm veriyi bir kez yükle
full_dataset = MyDatasetAll(data_dir, train_transform, seed=42)
N = len(full_dataset)
all_labels_array = np.array(full_dataset.labels)
print(f"Toplam örnek sayısı: {N}")

# ===================== MCCV MAIN LOOP =====================
for model_name in models_dict.keys():
    best_seed = seed_val[model_name]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Seed: {best_seed} | MCCV (K={K})")
    print(f"{'='*60}")

    # Her iterasyonun metriklerini topla
    iter_accuracies = []
    iter_precisions = []
    iter_recalls = []
    iter_f1s = []

    # Tüm iterasyonların tahminlerini topla (genel confusion matrix için)
    global_labels = []
    global_preds = []

    best_overall_f1 = 0.0
    best_overall_state = None

    # Stratified split: %70 train, %10 val, %20 test
    # İlk olarak %80 train+val / %20 test ayır, sonra train+val'i %87.5/%12.5 ayır (= toplam %70/%10)
    sss_outer = StratifiedShuffleSplit(n_splits=K, test_size=0.2, random_state=best_seed)

    for iter_idx, (trainval_indices, test_indices) in enumerate(sss_outer.split(np.zeros(N), all_labels_array)):

        seed_torch(seed=best_seed)

        # Train+Val'i ayır: %87.5 train, %12.5 val (toplam verinin %70 ve %10'u)
        trainval_labels = all_labels_array[trainval_indices]
        sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=best_seed + iter_idx)
        train_idx_local, val_idx_local = next(sss_inner.split(np.zeros(len(trainval_indices)), trainval_labels))

        train_indices = trainval_indices[train_idx_local]
        val_indices = trainval_indices[val_idx_local]

        train_subset = Subset(full_dataset, train_indices.tolist())
        val_subset = Subset(full_dataset, val_indices.tolist())
        test_subset = Subset(full_dataset, test_indices.tolist())

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        # Model oluştur
        input_size = 55
        hidden_size = 128
        output_size = 2
        cnn_featureNum = cnn_feature_map[model_name]

        mlp_branch_model = MLPBranch(input_size, hidden_size, output_size)
        attention_mdl = AttentionFusion(cnn_featureNum, 128)
        model = MultimodalModel(model_name, mlp_branch_model, attention_mdl, best_seed, device)
        model.to(device)

        params = list(model.cnn_branch.parameters()) + \
                 list(mlp_branch_model.parameters()) + \
                 list(attention_mdl.parameters())

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params, lr=0.001)

        # ---- Eğitim (validation + early stopping) ----
        best_val_f1 = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            model.train()
            for inputs, labels, text_inputs in tqdm(train_loader,
                    desc=f"  Iter {iter_idx+1}/{K} | Epoch {epoch+1}/{num_epochs}", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                text_inputs = text_inputs.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, text_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for inputs, labels, text_inputs in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float().unsqueeze(1)
                    text_inputs = text_inputs.to(device)
                    outputs = model(inputs, text_inputs)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                    val_labels.extend(labels.cpu().numpy().flatten().tolist())
                    val_preds.extend(preds.cpu().numpy().flatten().tolist())

            val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # ---- Test ----
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        iter_labels = []
        iter_preds = []
        with torch.no_grad():
            for inputs, labels, text_inputs in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                text_inputs = text_inputs.to(device)
                outputs = model(inputs, text_inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                iter_labels.extend(labels.cpu().numpy().flatten().tolist())
                iter_preds.extend(preds.cpu().numpy().flatten().tolist())

        # Bu iterasyonun metrikleri
        acc = accuracy_score(iter_labels, iter_preds)
        prec = precision_score(iter_labels, iter_preds, average='binary', zero_division=0)
        rec = recall_score(iter_labels, iter_preds, average='binary', zero_division=0)
        f1_val = f1_score(iter_labels, iter_preds, average='binary', zero_division=0)

        iter_accuracies.append(acc)
        iter_precisions.append(prec)
        iter_recalls.append(rec)
        iter_f1s.append(f1_val)

        global_labels.extend(iter_labels)
        global_preds.extend(iter_preds)

        # En iyi modeli sakla
        if f1_val > best_overall_f1:
            best_overall_f1 = f1_val
            best_overall_state = copy.deepcopy(best_state if best_state else model.state_dict())

        print(f"  Iter {iter_idx+1}/{K} | Acc: {acc:.4f} | F1: {f1_val:.4f} | Val F1: {best_val_f1:.4f}")

        del model, mlp_branch_model, attention_mdl
        torch.cuda.empty_cache()

    # ===================== MCCV SONUÇLARI =====================
    mean_acc = np.mean(iter_accuracies)
    std_acc = np.std(iter_accuracies)
    mean_prec = np.mean(iter_precisions)
    std_prec = np.std(iter_precisions)
    mean_rec = np.mean(iter_recalls)
    std_rec = np.std(iter_recalls)
    mean_f1 = np.mean(iter_f1s)
    std_f1 = np.std(iter_f1s)

    print(f'\nMCCV Results for {model_name} (seed={best_seed}, K={K}):')
    print(f'Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'Precision: {mean_prec:.4f} ± {std_prec:.4f}')
    print(f'Recall:    {mean_rec:.4f} ± {std_rec:.4f}')
    print(f'F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}')

    # Confusion Matrix (tüm iterasyonların toplamı)
    cm = confusion_matrix(global_labels, global_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'MCCV Confusion Matrix - {model_name}\nAcc={mean_acc:.3f}±{std_acc:.3f} | F1={mean_f1:.3f}±{std_f1:.3f}')
    plt.savefig(f'rev_mccv/mccv_{model_name}_cm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Metrik dağılımı box plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics_data = [iter_accuracies, iter_precisions, iter_recalls, iter_f1s]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for ax, data, name in zip(axes, metrics_data, metrics_names):
        ax.boxplot(data)
        ax.set_title(f'{name}\n{np.mean(data):.3f}±{np.std(data):.3f}')
        ax.set_ylim([0, 1])
    plt.suptitle(f'MCCV Metric Distributions - {model_name} (K={K})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'rev_mccv/mccv_{model_name}_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Sonuçları txt olarak kaydet
    result_txt = (
        f"Model: {model_name}\n"
        f"Seed: {best_seed}\n"
        f"Evaluation: Monte Carlo Cross-Validation (K={K})\n"
        f"Split: 70% train / 10% val / 20% test (stratified)\n"
        f"Epochs: {num_epochs} (early stopping, patience={patience})\n"
        f"---\n"
        f"Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"Precision: {mean_prec:.4f} ± {std_prec:.4f}\n"
        f"Recall:    {mean_rec:.4f} ± {std_rec:.4f}\n"
        f"F1-Score:  {mean_f1:.4f} ± {std_f1:.4f}\n"
        f"---\n"
        f"Per-iteration results:\n"
    )
    for i in range(K):
        result_txt += f"  Iter {i+1}: Acc={iter_accuracies[i]:.4f} Prec={iter_precisions[i]:.4f} Rec={iter_recalls[i]:.4f} F1={iter_f1s[i]:.4f}\n"

    with open(f"rev_mccv/mccv_{model_name}_result.txt", "w") as f:
        f.write(result_txt)

    # Best model kaydet
    if best_overall_state is not None:
        torch.save(best_overall_state, f'rev_mccv/mccv_{model_name}_best.pth')

    print(f"Saved: rev_mccv/mccv_{model_name}_cm.png")
    print(f"Saved: rev_mccv/mccv_{model_name}_boxplot.png")
    print(f"Saved: rev_mccv/mccv_{model_name}_result.txt")
    print(f"Saved: rev_mccv/mccv_{model_name}_best.pth")