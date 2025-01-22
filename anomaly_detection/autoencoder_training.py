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
from sklearn.model_selection import train_test_split
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
def plot_history(history):
    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))

    plt.plot(history['loss'], label='Train Loss')
    if any(val is not None for val in history['val_loss']):
        val_loss = [val if val is not None else float('nan') for val in history['val_loss']]
        plt.plot(val_loss, label='Val Loss', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('AD_19-23_autoencoder_loss_noLapTime.png')
    plt.show()


    plt.figure(figsize=(12, 6))

    plt.plot(history['thresholds99_9'], label='Thresholds')
    plt.xlabel('Epochs')
    plt.ylabel('Threshold')
    plt.title('Thresholds')
    plt.legend()
    plt.grid(True)
    plt.savefig('AD_19-23_autoencoder_thresholds_noLapTime.png')
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


def train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, device, learning_rate, criterion,
                      optimizer_name='Adam'):
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

    history = {"loss": [], "val_loss": [], "reconstruction_error": [], 'thresholds99_9': []}

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["loss"].append(train_loss)

        if epoch % validation_freq == 0 or epoch == epochs or epoch == 1:
            autoencoder.eval()
            val_loss = 0
            val_reconstruction_error = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = autoencoder(batch)

                    loss = criterion(outputs, batch)
                    val_loss += loss.item()

                    error = torch.mean(torch.abs(outputs - batch)).item()
                    val_reconstruction_error.append(error)

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            # Calculate the percentile thresholds from reconstruction errors
            thresholds = np.percentile(val_reconstruction_error, [95, 99, 99.5, 99.9])

            history["thresholds99_9"].append(thresholds[-1])
            history["reconstruction_error"].append(val_reconstruction_error)

            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"Calculated percentile thresholds:\n"
                f"95th: {thresholds[0]:.4f}\n"
                f"99th: {thresholds[1]:.4f}\n"
                f"99.5th: {thresholds[2]:.4f}\n"
                f"99.9th: {thresholds[3]:.4f}\n")
        else:
            history["val_loss"].append(None)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}\n")

        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary(device=None, abbreviated=True))

        # Save the trained model
        print("Saving the trained model...")
        learning_rate_str = str(learning_rate).split(".")[1]
        torch.save(autoencoder.state_dict(),
                   f"saved_models/v4_noLapTime/AD_19-23_autoencoder_{optimizer_name}_lr{learning_rate_str}_ep{epoch}_loss{train_loss:.4f}.pth")
        print("Model saved.")



    print(

    )
    return history


# Main function
if __name__ == "__main__":
    # Clear the output
    os.system('cls' if os.name == 'nt' else 'clear')

    # Check for CUDA/GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parameters
    sequence_length = 20
    batch_size = 128
    epochs = 15
    validation_freq = 3
    learning_rate = 0.0001
    optimizer_name = 'AdamW'
    criterion = nn.L1Loss()

    # Path to the dataset
    dataset_path = "D:/F1LLM_Datasets/npz_normalized/train_data/train_data_without_failures"

    # Load data
    data = load_data(dataset_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create datasets and loaders
    train_dataset = FastF1Dataset(train_data, sequence_length)
    val_dataset = FastF1Dataset(val_data, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the autoencoder
    autoencoder = LSTMAutoencoder(sequence_length, data.shape[1]).to(device)

    history = train_autoencoder(autoencoder,
                                train_loader,
                                val_loader,
                                epochs,
                                validation_freq,
                                device,
                                learning_rate,
                                criterion,
                                optimizer_name)

    plot_history(history)