import pandas as pd
import numpy as np

data = pd.read_excel('data_old.xlsx')

unique_values = set()

# Iterate through the 'eko bulguları' column and collect unique values
for i in list(data['eko bulguları']):
    if type(i) == float: continue
    for j in i.split(','):
        unique_values.add(j.strip())

# Create a mapping of unique values to indices
# The mapping will be a dictionary where the key is the unique value and the value is its index
mapping = {v:idx for idx, v in enumerate(list(unique_values))}

data['NT proBNP'] = (data['NT proBNP'] - data['NT proBNP'].mean()) / data['NT proBNP'].std()

# Initialize a dictionary to hold the train, validation, and test splits
train_val_test_features = {"train": [], "val": [], "test": []}
train_val_test_labels = {"train": [], "val": [], "test": []}

# Iterate through the DataFrame rows and populate the train_val_test dictionary
for _, row in data.iterrows():
    idx = []

    if type(row['eko bulguları']) != float:
        # Split the string by commas and map each value to its corresponding index
        for i in row['eko bulguları'].split(','):
            if type(i) == float: continue
            idx.append(mapping[i.strip()])

    # Create a zero-initialized array of length equal to the number of unique values + 4 for the additional features
    # and set the corresponding indices to 1
    record = np.zeros(len(mapping) + 4)
    record[idx] = 1
 
    record[-4:] = [row['kreatinin'], row['NT proBNP'] if row['NT proBNP'] != 'yok' else 0, row['eko EF'], row['3. SAAT K/KL']]
    
    train_val_test_features[row['Split']].append(record.tolist())
    train_val_test_labels[row['Split']].append(row['PYP SEMİKANTİTATİF'])

# Create and fit RandomForestClassifier with train_val_test['train']
X_train = np.array(train_val_test_features['train'])
y_train = np.array(train_val_test_labels['train'])

# Predict on the validation set
X_val = np.array(train_val_test_features['val'])
y_val = np.array(train_val_test_labels['val'])

# Predict on the test set
X_test = np.array(train_val_test_features['test'])
y_test = np.array(train_val_test_labels['test'])

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.output(x)

# Model parameters
input_size = X_train.shape[1]
hidden_size = 128
output_size = len(set(y_train))  # Number of unique labels

# Initialize the model, loss function, and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Training loop
num_epochs = 200
for epoch in range(num_epochs):

    # Training step with DataLoader
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation step with DataLoader
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            val_outputs = model(batch_X)
            val_predictions.extend(torch.argmax(val_outputs, dim=1).numpy())
            val_labels.extend(batch_y.numpy())

    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if epoch == 0 or val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")

# Load the best model for evaluation on the test set
model.load_state_dict(torch.load("best_model.pth"))

# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = torch.argmax(test_outputs, dim=1)
    test_accuracy = accuracy_score(y_test, test_predictions.numpy())

# Print the test accuracy and classification report
from sklearn.metrics import classification_report
print("Test Accuracy:", test_accuracy)
print(classification_report(y_test, test_predictions.numpy()))

print(f"Test Accuracy: {test_accuracy:.4f}")