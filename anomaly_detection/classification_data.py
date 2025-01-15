import numpy as np
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from data_extraction_and_preprocessing.normalize_npz import Normalizer
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
    return torch.mean((inputs - outputs) ** 2, dim=1).detach().numpy()


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


def plot_reconstruction_errors_with_threshold(errors, threshold=None):
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
    plt.xticks(range(0, len(errors), 300))
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Errors with Threshold")

    # Aggiungi legenda una sola volta
    # plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    #save
    # plt.savefig('reconstruction_errors.png')
    plt.show()


def extract_last_laps(df, failures_np):
    print(df.shape, failures_np.shape)
    model_path = "AD_19-23_autoencoder_AdamW_lr0001_loss0.4037_fold5.pth"
    sequence_length = 20

    test_dataset = FastF1Dataset(df, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = df.shape[1]  # Number of features per timestep
    autoencoder = load_model(model_path, LSTMAutoencoder, sequence_length, input_dim, device).to(device)

    reconstruction_errors = test_autoencoder(autoencoder, test_loader, device)

    threshold = np.percentile(reconstruction_errors, 99.9)
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
    plot_reconstruction_errors_with_threshold(reconstruction_errors, threshold=threshold)
    #add failures_df column to the df
    failures_np = failures_np.reshape(-1, 1)
    recombined_array = np.hstack((df, failures_np))

    return recombined_array[last_anomaly:]


# main
if __name__ == "__main__":

    # Normalize the data
    normClass = Normalizer(pit_stops=True)

    folder_path = '../Dataset/OnlyFailuresByDriver/npz_failures'
    train_mode = False

    driver = 23  # set a value or None
    event = "SingaporeGrandPrix"  # set a value or None

    df_columns = [
        'Time_x', 'Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime',
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
        'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'Compound_x', 'TyreLife_x', 'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate', 'TrackStatus',
        'Position', 'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate', 'Compound_y',
        'TyreLife_y', 'TimeXY', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
        'WindDirection', 'WindSpeed', 'Date', 'SessionTime', 'DriverAhead', 'DistanceToDriverAhead',
        'Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source', 'Distance',
        'RelativeDistance', 'Status', 'X', 'Y', 'Z', 'Year', 'Event'
    ]

    all_data = []
    for f in os.listdir(folder_path):

        data = None

        if train_mode and f.endswith('.npz') and not f.startswith('2024'):
            if driver is None and event is None:
                print(f'Loading {f}...')
                data = np.load(os.path.join(folder_path, f), allow_pickle=True)['data']
            else:
                print('You specified a driver and/or an event in \'train\' mode.')
                break
        elif not train_mode and f.endswith('.npz') and f.startswith('2024'):
            if driver is None and event is None:
                print(f'Loading {f}...')
                data = np.load(os.path.join(folder_path, f), allow_pickle=True)['data']
            else:
                driver_file = int(f.split('_')[2])
                event_file = f.split('_')[1]
                # print(f'Driver: {driver_file}, Event: {event_file}')
                if driver == driver_file and event == event_file:
                    print(f'Loading {f}...')
                    data = np.load(os.path.join(folder_path, f), allow_pickle=True)['data']
                else:
                    continue

        if data is not None and data.size > 0:
            df = pd.DataFrame(data, columns=df_columns)


            # Ensure the DataFrame is not empty
            if df.empty:
                print(f"DataFrame for file {f} is empty. Skipping this file.")
                continue

            # Add the failure column
            failure = f.split('_')[3].split(".")[0]
            df['Failure'] = failure
            norm_np = normClass.normalize_data(df)

            failure_np = norm_np[:, -1]
            #drop last column of norm_np
            norm_np = norm_np[:, :-1]

            norm_np = extract_last_laps(norm_np, failure_np)

            # norm_df = normClass.normalize_data(df)
            all_data.append(norm_np)
            print(f'Shape: {norm_np.shape}')
            print(f'Loaded {f}!, All data: {len(all_data)}')
        else:
            # print(f"File {f} did not contain valid data.")
            continue

    if len(all_data) > 1:
        norm_final_failure = np.concatenate(all_data, axis=0)
    else:
        norm_final_failure = all_data[0]

    # Save the dataset based on the mode (train or test)
    dataset_type = "train" if train_mode else "test"
    if driver is not None and event is not None:
        output_file = f'../Dataset/OnlyFailuresByDriver/npz_failures_MinMaxScaler_normalized_{dataset_type}_{driver}_{event}.npz'
    else:
        output_file = f'../Dataset/OnlyFailuresByDriver/npz_failures_MinMaxScaler_normalized_{dataset_type}.npz'
    np.savez_compressed(output_file, data=norm_final_failure)

    print(f"Dataset saved as: {output_file}")
