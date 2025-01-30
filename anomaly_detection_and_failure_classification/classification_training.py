import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import os


# Custom Failures with Dynamic Sliding Window
class FailureDataset(Dataset):
    def __init__(self, race_data, sequence_length):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length

        # Process each race independently
        for race in race_data:
            race_features = torch.tensor(race[:, :-1], dtype=torch.float32)
            race_labels = torch.tensor(race[:, -1], dtype=torch.long)

            # Create sliding windows within the race
            for i in range(len(race_features) - sequence_length + 1):
                # end = min(self.sequence_length, len(race_features) - i)
                self.data.append(race_features[i:i + sequence_length])
                self.labels.append(race_labels[i + sequence_length - 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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


def main():
    # Load the dataset
    new_data_path = "D:/F1LLM_Datasets/OnlyFailuresByDriver/npz_failures_MinMaxScaler_normalized_train.npz"
    data = np.load(new_data_path, allow_pickle=True)['data']

    # Separate features and labels
    X = data[:, :-1]  # All columns except the last one (features)
    y = data[:, -1].astype(int)  # The last column as integer labels (target)

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.0001
    epochs = 5
    sequence_length = 100
    n_features = X.shape[1]  # Number of features per sample
    n_classes = len(np.unique(y))  # Automatically determine the number of classes

    # Split the data into team-race combinations
    team_race_ids = np.array([f"{team}-{race}" for team, race in zip(data[:, -8], data[:, -2])])

    # Get unique team-race combinations

    unique_team_race_ids = np.unique(team_race_ids)
    print(len(team_race_ids), len(unique_team_race_ids))
    print(unique_team_race_ids)

    # Set up K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize the model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMModel(n_features=n_features, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Directory to save models
    model_save_dir = "models"
    os.makedirs(model_save_dir, exist_ok=True)

    # K-Fold Cross Validation Loop
    for fold, (train_indices, val_indices) in enumerate(kf.split(unique_team_race_ids)):
        print(f"Fold {fold + 1}/{kf.get_n_splits()}")

        # Split the team-race IDs for this fold
        train_team_race_ids = unique_team_race_ids[train_indices]
        val_team_race_ids = unique_team_race_ids[val_indices]

        # Create masks to filter the data for training and validation
        train_mask = np.isin(team_race_ids, train_team_race_ids)
        val_mask = np.isin(team_race_ids, val_team_race_ids)

        # Split the data into training and validation sets
        train_data = data[train_mask]
        val_data = data[val_mask]

        # Process each race independently for the FailureDataset
        train_races = [train_data[train_data[:, -2] == race_id] for race_id in np.unique(train_data[:, -2])]
        val_races = [val_data[val_data[:, -2] == race_id] for race_id in np.unique(val_data[:, -2])]

        # Create datasets
        train_dataset = FailureDataset(train_races, sequence_length)
        val_dataset = FailureDataset(val_races, sequence_length)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            model_save_path = os.path.join(model_save_dir, f"classification_model_sequence500_fold{fold + 1}_epoch{epoch + 1}_loss{epoch_loss:.4f}.pth")
            torch.save(model.state_dict(), model_save_path)

        # Evaluate after each fold
        accuracy = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy after Fold {fold + 1}: {accuracy:.2f}%")

    # Final evaluation after all folds
    print("K-Fold Cross Validation Complete")



if __name__ == "__main__":
    main()
