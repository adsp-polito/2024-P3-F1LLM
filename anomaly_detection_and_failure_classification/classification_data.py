import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class FastF1Dataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx + self.sequence_length], dtype=torch.float32)


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


def load_model(model_path, model_class, sequence_length, input_dim, device):
    model = model_class(sequence_length, input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def compute_reconstruction_error(inputs, outputs):
    return torch.mean((inputs - outputs) ** 2, dim=1).detach().cpu().numpy()


def test_autoencoder(autoencoder, data_loader, device):
    autoencoder.eval()
    errors = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.to(device)
            outputs = autoencoder(inputs)
            batch_errors = compute_reconstruction_error(inputs, outputs)
            errors.extend(batch_errors)

    return np.array(errors)


def plot_reconstruction_errors_with_threshold(errors, threshold=None, event_name=None):
    """
    Plots reconstruction errors as a line graph with an optional threshold.

    Parameters:
        errors (np.array): Array of reconstruction errors.
        threshold (float, optional): Threshold for detecting anomalies. Defaults to None.
    """
    plt.figure(figsize=(24, 6))
    plt.plot(errors, label="Reconstruction Errors", color='blue', linewidth=1.5)

    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")

    plt.xlabel("Sample Index")
    plt.xticks(range(0, len(errors), 1000))
    plt.ylabel("Reconstruction Error")
    plt.title(f"{event_name} - Reconstruction Errors with Threshold")

    # Aggiungi legenda una sola volta
    plt.grid(alpha=0.3)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()


def extract_last_laps(df, failures_np, event_name=None):
    print(df.shape, failures_np.shape)
    model_path = "models/autoencoder_AdamW_lr0001_loss0.4037_fold5.pth"
    sequence_length = 20

    test_dataset = FastF1Dataset(df, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = df.shape[1]  # Number of features per timestep
    autoencoder = load_model(model_path, LSTMAutoencoder, sequence_length, input_dim, device).to(device)

    reconstruction_errors = test_autoencoder(autoencoder, test_loader, device)

    threshold = 0.2061
    anomalies = reconstruction_errors > threshold

    indexes = np.where(anomalies == True)[0]

    # Group consecutive True blocks, allowing gaps of up to 200
    blocks = []
    block = []
    for i in range(len(indexes)):
        if not block or indexes[i] <= block[-1] + 200:
            # Start a new block or extend the current one
            block.append(indexes[i])
        else:
            blocks.append(block)
            block = [indexes[i]]

    # Append the last block
    if block:
        blocks.append(block)

    last_anomaly = blocks[-1][0] - 200
    print(f"Last anomaly detected at index {last_anomaly}.")
    plot_reconstruction_errors_with_threshold(reconstruction_errors, threshold=threshold, event_name=event_name)

    # add failures_df column to the df
    failures_np = failures_np.reshape(-1, 1)
    recombined_array = np.hstack((df, failures_np))

    return recombined_array[last_anomaly:]


# main
if __name__ == "__main__":

    input_folder_path = 'D:/F1LLM_Datasets/npz_normalized/train_data/train_data_only_failures'
    output_folder_path = 'D:/F1LLM_Datasets/npz_normalized_last_anomaly_extraction'

    for f in os.listdir(input_folder_path):
        print(f'Loading {f}...')
        data = np.load(os.path.join(input_folder_path, f), allow_pickle=True)['data']

        if data.size > 0:

            failure_np = data[:, -1]
            data = data[:, :-1]

            data = extract_last_laps(data, failure_np, event_name=f.split('.')[0].split('_')[1])

            print(f'Shape: {data.shape}')

            output_file = f.split('.')[0] + '_last_anomaly.npz'
            np.savez_compressed(os.path.join(output_folder_path, output_file), data=data)

            print(f"Dataset saved as: {output_file}")
        else:
            print(f"File {f} did not contain valid data.")
            continue
