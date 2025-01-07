import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os  # For file management

# Custom Dataset with Dynamic Sliding Window
class FailureDataset(Dataset):
    """
    Dataset for dynamically generating data samples using a sliding window approach.
    Each sample is a window of fixed length (`sequence_length`), and the label corresponds
    to the last timestep in the window.
    """
    def __init__(self, data, labels, sequence_length):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert data to PyTorch tensors
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to PyTorch tensors
        self.sequence_length = sequence_length  # Length of the sliding window

    def __len__(self):
        # Number of possible windows = total samples - window size + 1
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Extract the sliding window starting at index `idx`
        sample = self.data[idx:idx + self.sequence_length]
        # The label corresponds to the last timestep in the window
        label = self.labels[idx + self.sequence_length - 1]
        return sample, label


# Model Definition
class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for classification.
    Combines convolutional layers for feature extraction and LSTM layers for sequential modeling.
    """
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
        # CNN layers
        x = x.permute(0, 2, 1)  # Switch to (batch_size, n_features, sequence_length)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # LSTM layers
        x = x.permute(0, 2, 1)  # Switch back to (batch_size, sequence_length, features)
        x, _ = self.lstm(x)

        # Fully connected layers
        x = x[:, -1, :]  # Use the last LSTM output
        x = self.fc(x)
        return x


# Evaluation function
def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.
    Computes accuracy and prints the results.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# Main function
def main():
    # Load the dataset
    new_data_path = "Dataset/OnlyFailuresByDriver/npz_failures_MinMaxScaler_normalized_train.npz"
    data = np.load(new_data_path, allow_pickle=True)['data']

    # Separate features and labels
    X = data[:, :-1]  # All columns except the last one (features)
    y = data[:, -1].astype(int)  # The last column as integer labels (target)

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    epochs = 20
    sequence_length = 20  # Length of the sliding window
    n_features = X.shape[1]           # Number of features per sample
    n_classes = len(np.unique(y))     # Automatically determine the number of classes

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create DataLoaders with FailureDataset
    train_dataset = FailureDataset(X_train, y_train, sequence_length)
    test_dataset = FailureDataset(X_test, y_test, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMModel(n_features=n_features, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Directory to save models
    model_save_dir = "classification_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # Training Loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save the model with the epoch loss in the filename
        model_save_path = os.path.join(model_save_dir, f"classification_model_epoch{epoch+1}_loss{epoch_loss:.4f}.pth")
        torch.save(model.state_dict(), model_save_path)

        # Perform evaluation after the first epoch and every 5 epochs
        if epoch == 0 or (epoch + 1) % 5 == 0:
            evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()