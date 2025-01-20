import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import os  # For file management

from test import input_folder_path


# Custom Dataset with Dynamic Sliding Window
class FailureDataset(Dataset):
    def __init__(self, data, labels, sequence_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        sample = self.data[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length - 1]
        return sample, label


# Model Definition
class CNNLSTMModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, n_features, sequence_length)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, features)
        x, _ = self.lstm(x)

        x = x[:, -1, :]  # Use the last LSTM output
        x = self.fc(x)
        return x


# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Main function with K-Fold Cross Validation
def main():

    # Load the dataset
    input_folder_path = "D:\F1LLM_Datasets/npz_normalized/train_data/train_data_only_failures"

    all_data = []
    for file in os.listdir(input_folder_path):
        if file.endswith(".npz") and not file.startswith("2024"):
            file_path = os.path.join(input_folder_path, file)
            data = np.load(file_path, allow_pickle=True)['data']

            all_data.append(data)

    data = np.concatenate(all_data, axis=0)

    # Separate features and labels
    X = data[:, :-1]  # All columns except the last one (features)
    y = data[:, -1].astype(int)  # The last column as integer labels (target)

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    epochs = 20
    sequence_length = 20
    n_features = X.shape[1]  # Number of features per sample
    n_classes = len(np.unique(y))  # Automatically determine the number of classes
    k_folds = 5  # Number of splits for K-Fold Cross-Validation

    # K-Fold Cross Validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directory to save models
    model_save_dir = "classification_models"
    os.makedirs(model_save_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split the data into training and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create DataLoaders for the current fold
        train_dataset = FailureDataset(X_train, y_train, sequence_length)
        val_dataset = FailureDataset(X_val, y_val, sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, loss, and optimizer
        model = CNNLSTMModel(n_features=n_features, n_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # You can change optimizer here

        # Training Loop
        for epoch in range(epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Calculate average loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

            # Save the model after each epoch
            model_save_path = os.path.join(model_save_dir,
                                           f"classification_model_fold{fold + 1}_epoch{epoch + 1}_loss{epoch_loss:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)

            # Evaluate after each epoch
            if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
                accuracy = evaluate_model(model, val_loader, device)
                print(f"Validation Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%")

        # Evaluate the model on the final epoch of this fold
        final_accuracy = evaluate_model(model, val_loader, device)
        print(f"Final Validation Accuracy for Fold {fold + 1}: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
