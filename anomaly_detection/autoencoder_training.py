import time
import numpy as np

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm


# Updated function to load and preprocess data
def load_data(folder_path):
    """
    Loads FastF1 data and preprocesses it for sequence modeling.
    Args:
        folder_path (str): Path to the dataset folder.
    Returns:
        array: Preprocessed dataset ready for sequence modeling.
    """

    # Load dataset
    print(f'Loading dataset from {folder_path}...')
    all_data = []
    start_time = time.time()
    for file in os.listdir(folder_path):
        if file.endswith('.npz') and not file.startswith('2024'):
            print(f'Loading {file}...')
            stime = time.time()
            file_path = os.path.join(folder_path, file)
            np_data = np.load(file_path, allow_pickle=True)['data']
            print(f'Loaded! Appending to list...')
            all_data.append(np_data)
            etime = time.time()
            print(f'Done in {etime - stime:.2f} seconds')
    
    print(f'Concatenating data...')
    np_data = np.concatenate(all_data, axis=0)
  
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Dataset loaded in: {elapsed_time:.2f} seconds')
    
    return np_data

# Function to plot histories
def plot_histories(histories_list):
    """
    Plots the training and validation loss for each fold.

    Args:
        histories_list (list): List of history dictionaries from each fold.
    """
    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))
    for fold, history in enumerate(histories_list):
        plt.plot(history['loss'], label=f'Fold {fold + 1} Train Loss')
        if any(val is not None for val in history['val_loss']):
            val_loss = [val if val is not None else float('nan') for val in history['val_loss']]
            plt.plot(val_loss, label=f'Fold {fold + 1} Val Loss', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Each Fold')
    plt.legend()
    plt.grid(True)
    plt.savefig('AD_19-23_autoencoder_loss_noLapTime.png')
    plt.show()

class FastF1Dataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.sequence_length]

class LSTMAutoencoder(nn.Module):
    def __init__(self, sequence_length, num_features):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(num_features, 64, batch_first=True)
        self.latent = nn.LSTM(64, 32, batch_first=True)
        self.decoder = nn.LSTM(32, 64, batch_first=True)
        self.output_layer = nn.Linear(64, num_features)

    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.latent(x[:, -1].unsqueeze(1).repeat(1, x.size(1), 1))
        x, _ = self.decoder(x)
        x = self.output_layer(x)
        return x


def train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, device, learning_rate, criterion, optimizer_name='Adam', scheduler_name='StepLR', step_size=10, gamma=0.1):
    scaler = GradScaler('cuda')

    # Set up optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(autoencoder.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer name. Use 'Adam', 'AdamW', or 'SGD'.")

    history = {"loss": [], "val_loss": [], "reconstruction_error": [], "threshold95": None, "threshold99": None, "threshold99_5": None, "threshold99_9": None}
    reconstruction_errors = []

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        train_loss = 0
        reconstruction_error = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)

            # Compute MAE for reconstruction error
            mae = torch.mean(torch.abs(outputs - batch)).item()
            reconstruction_error += mae
            reconstruction_errors.append(mae)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["loss"].append(train_loss)
        reconstruction_error /= len(train_loader)
        history["reconstruction_error"].append(reconstruction_error)

        if epoch % validation_freq == 0 or epoch == epochs or epoch == 1:
            autoencoder.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = autoencoder(batch)
                    loss = criterion(outputs, batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Recon Error: {reconstruction_error:.4f}")
        else:
            history["val_loss"].append(None)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Recon Error: {reconstruction_error:.4f}")

        # Libera la memoria non utilizzata
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=None, abbreviated=True))

        # Save the trained model
        print("Saving the trained model...")
        learning_rate_str = str(learning_rate).split(".")[1]
        torch.save(autoencoder.state_dict(), f"saved_models/v4_noLapTime/AD_19-23_autoencoder_{optimizer_name}_lr{learning_rate_str}_ep{epoch}_loss{train_loss:.4f}.pth")
        print("Model saved.")

    # Calculate the 95th percentile threshold from reconstruction errors
    threshold95 = np.percentile(reconstruction_errors, 95)
    threshold99 = np.percentile(reconstruction_errors, 99)
    threshold99_5 = np.percentile(reconstruction_errors, 99.5)
    threshold99_9 = np.percentile(reconstruction_errors, 99.9)
    history["threshold95"] = threshold95
    history["threshold99"] = threshold99
    history["threshold99_5"] = threshold99_5
    history["threshold99_9"] = threshold99_9
    print(f"Calculated percentile thresholds:\n95th: {threshold95:.4f}, 99th: {threshold99:.4f}, 99.5th: {threshold99_5:.4f}, 99.9th: {threshold99_9:.4f}")

    return history


def cross_validate_autoencoder(autoencoder_class, data, k_folds, sequence_length, batch_size, epochs, 
                               validation_freq, device, learning_rate, criterion, optimizer_name):
    """
    Performs k-fold cross-validation for the LSTM Autoencoder.
    
    Args:
        autoencoder_class (class): Class of the autoencoder model.
        data (ndarray): Full dataset to split into folds.
        k_folds (int): Number of folds for cross-validation.
        sequence_length (int): Sequence length for the model.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for each fold.
        validation_freq (int): Frequency of validation.
        device (torch.device): Device to use for training.
        learning_rate (float): Learning rate for the optimizer.
        criterion (loss): Loss function.
        optimizer_name (str): Optimizer to use ("Adam", "AdamW", etc.).
    
    Returns:
        list: Training histories for each fold.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    histories_list = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(data)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---\n")
        
        # Split the data
        train_data = data[train_indices]
        val_data = data[val_indices]
        
        # Create datasets and loaders
        train_dataset = FastF1Dataset(train_data, sequence_length)
        val_dataset = FastF1Dataset(val_data, sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize a new autoencoder for each fold
        autoencoder = autoencoder_class(sequence_length, data.shape[1]).to(device)
        
        # Train the autoencoder on the current fold
        history = train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, 
                                    device, learning_rate, criterion, optimizer_name)
        
        # Store the training history
        histories_list.append(history)

        learning_rate_str = str(learning_rate).split(".")[1]

        # Optionally, save the model for each fold
        torch.save(autoencoder.state_dict(), f"saved_models/v4_noLapTime/AD_19-23_autoencoder_{optimizer_name}_lr{learning_rate_str}_loss{history['loss'][-1]:.4f}_fold{fold + 1}.pth")
        print(f"Model for fold {fold + 1} saved.")
    
    return histories_list

# Main function
if __name__ == "__main__":
    # Clear the output
    os.system('cls' if os.name == 'nt' else 'clear')

    # Check for CUDA/GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Path to the dataset
    dataset_path = "D:/F1LLM_Datasets/npz_normalized/train_data/train_data_without_failures"

    # Load and preprocess data
    data = load_data(dataset_path)

    # Parameters
    sequence_length = 20
    batch_size = 128
    epochs = 10
    validation_freq = 5
    learning_rate = 0.0001
    optimizer_name = 'AdamW'
    criterion = nn.L1Loss()
    k_folds = 5  # Number of cross-validation folds

    # Run k-fold cross-validation
    histories = cross_validate_autoencoder(LSTMAutoencoder, data, k_folds, sequence_length, 
                                           batch_size, epochs, validation_freq, device, 
                                           learning_rate, criterion, optimizer_name)
    
    plot_histories(histories)