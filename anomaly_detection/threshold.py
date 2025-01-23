import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn

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

def load_data(folder_path):
    # Load dataset
    print(f'Loading dataset from {folder_path}...')
    all_data = []
    start_time = time.time()
    for file in os.listdir(folder_path):
        if file.endswith('.npz') and not file.startswith('2024'):
            print(f'Loading {file}...')
            file_path = os.path.join(folder_path, file)
            np_data = np.load(file_path, allow_pickle=True)['data']
            print(f'Loaded! Appending to list...')
            all_data.append(np_data)


    print(f'Concatenating data...')
    np_data = np.concatenate(all_data, axis=0)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Dataset loaded in: {elapsed_time:.2f} seconds')

    return np_data

def evaluate_autoencoder(autoencoder, val_loader, device, model_path):
    # Load the trained model weights
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()

    # Initialize list to store reconstruction errors
    reconstruction_errors = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Autoencoder"):
            batch = batch.to(device)
            outputs = autoencoder(batch)

            # Compute reconstruction error for the batch
            errors = compute_reconstruction_error(batch, outputs)
            reconstruction_errors.extend(errors)  # Append errors to the list

    return reconstruction_errors

def compute_reconstruction_error(inputs, outputs):
    return torch.mean((inputs - outputs) ** 2, dim=(1, 2)).detach().cpu().numpy()

if __name__ == "__main__":
    # Parameters
    sequence_length = 20
    batch_size = 128
    epochs = 15
    validation_freq = 3
    learning_rate = 0.0001
    optimizer_name = 'AdamW'
    criterion = nn.L1Loss()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "saved_models/v4_noLapTime/AD_19-23_autoencoder_AdamW_lr0001_ep15_loss0.5003.pth"  # Replace with your model path

    dataset_path = "D:/F1LLM_Datasets/npz_normalized/train_data/train_data_without_failures"
    data = load_data(dataset_path)
    autoencoder = LSTMAutoencoder(sequence_length, data.shape[1]).to(device)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    # Create datasets and loaders
    train_dataset = FastF1Dataset(train_data, sequence_length)
    val_dataset = FastF1Dataset(val_data, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    reconstruction_errors = evaluate_autoencoder(autoencoder, val_loader, device, model_path)

    thresholds = np.percentile(reconstruction_errors, [95, 99, 99.5, 99.9])
    print("Thresholds:", thresholds)