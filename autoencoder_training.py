import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


def convert_time_to_seconds(time_str):
    """
    Converts a timestamp string (e.g., '00:00.557') to seconds as float.
    Args:
        time_str (str): Time string in 'HH:MM:SS.sss' or 'MM:SS.sss' format.
    Returns:
        float: Time in seconds.
    """
    try:
        if pd.isnull(time_str) or time_str.strip() == '':
            return 0.0
        parts = time_str.split(':')
        if len(parts) == 3:  # HH:MM:SS.sss
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:  # MM:SS.sss
            m, s = map(float, parts)
            return m * 60 + s
        else:
            return float(time_str)  # Directly convert if already numeric
    except ValueError:
        return 0.0  # Handle invalid format gracefully

# Check which columns are present in the dataset
def check_columns(df, columns):
    """
    Args:
        df (DataFrame): Input dataset.
        columns (list): List of columns to check.
    Returns:
        list: Columns present in the dataset.
    """
    return [col for col in columns if col in df.columns]

def map_grand_prix(df):
    """
    Maps Grand Prix events to unique numeric identifiers and removes the original 'Event' column.
    
    Args:
        df (DataFrame): Input DataFrame containing the 'Event' column.
    
    Returns:
        DataFrame: Updated DataFrame with the new column 'Event_mapped'.
        dict: Mapping of events to their numeric identifiers.
    """
    # Create a sorted list of unique events
    grand_prix_list = sorted(df['Event'].unique())  # Ensure consistent mapping order
    event_mapping = {event: idx + 1 for idx, event in enumerate(grand_prix_list)}  # Map to integers

    # Add the mapped column
    df['Event_mapped'] = df['Event'].map(event_mapping).fillna(0).astype(int)
    
    # Drop the original 'Event' column
    df = df.drop(columns=['Event'], errors='ignore')
    
    return df, event_mapping


def map_team(df):
    """
    Maps teams to unique numeric identifiers and removes the original 'Team' column.
    
    Args:
        df (DataFrame): Input DataFrame containing the 'Team' column.
    
    Returns:
        DataFrame: Updated DataFrame with the new column 'Team_mapped'.
        dict: Mapping of events to their numeric identifiers.
    """
    # Create a sorted list of unique events
    team_list = sorted(df['Team'].unique())  # Ensure consistent mapping order
    team_mapping = {team: idx + 1 for idx, team in enumerate(team_list)}  # Map to integers

    # Add the mapped column
    df['Team_mapped'] = df['Team'].map(team_mapping).fillna(0).astype(int)
    
    # Drop the original 'Team' column
    df = df.drop(columns=['Team'], errors='ignore')
    
    return df, team_mapping


# Preprocess FastF1 data specifications
def normalize_data(df):
    """
    Features are handled as follows:
    1. Numerical features: Standardized using `StandardScaler` to ensure all values are on the same scale.
       - Examples: Lap times, sector times, environmental variables, telemetry data, etc.

    2. Categorical features: One-hot encoded using `OneHotEncoder` to create binary vector representations.
       - Examples: Tire compound, track status.

    3. Pit stop time: Calculated as the difference between `PitOutTime` and `PitInTime`, then normalized.

    4. Non-normalized features: Retained as-is since they are either already in the correct format or represent discrete values.
       - Examples: Driver number, stint number, lap number.

    5. Dropped features: Irrelevant or redundant features removed based on domain knowledge (for this kind of training).
       - Examples: Driver name, team name, inaccurate or redundant telemetry features.
    
    Args:
        df (DataFrame): Input dataset containing raw telemetry and race data.
    
    Returns:
        np.array: Preprocessed dataset, combining standardized numerical features, 
                  one-hot encoded categorical features, and unprocessed features.
    """
    
    print('Preprocessing data...')
    start_time = time.time()

    # List of time columns to convert
    time_columns = [
        'Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'PitOutTime', 'PitInTime', 'TimeXY'
    ]

    # Convert all time columns to seconds
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_time_to_seconds)

    df['DriverAhead'] = df['DriverAhead'].apply(lambda x: x if pd.isna(x) else int(x))

    # Columns to normalize
    numerical_cols = [
        'Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'Speed', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
        'WindDirection', 'WindSpeed', 'DistanceToDriverAhead', 'RPM', 'nGear',
        'Throttle', 'Brake', 'DRS', 'X', 'Y', 'Z', 'Distance'
    ]

    # THIS IS TEMPORARY: WE NEED TO MANAGE BETTER THE "OBJECT" TYPE

    # Handle `Compound` with OneHotEncoder
    df.rename(columns={'Compound_x': 'Compound'}, inplace=True)

    if 'Compound' in df.columns:
        one_hot = pd.get_dummies(df['Compound'], prefix='Compound')
        df = pd.concat([df, one_hot], axis=1).drop(columns=['Compound'])
    
    # Map Team column to numeric identifiers
    if 'Team' in df.columns:
        df, team_mapping = map_team(df)
        # print("Mapped Teams:", team_mapping)

    # Map Event column to numeric identifiers
    if 'Event' in df.columns:
        df, event_mapping = map_grand_prix(df)
        # print("Mapped Grand Prix events:", event_mapping)

    # Columns to one-hot encode
    categorical_cols = ['TrackStatus']

    # Columns to leave unprocessed
    non_normalized_cols = ['DriverNumber', 'Stint', 'LapNumber', 'Position', 'IsPersonalBest', 'Year', 'Event', 'Team']

    # Columns to drop (irrelevant or redundant)
    drop_cols = [
        'Driver', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreLife_x', 'FreshTyre', 'LapStartTime', 'LapStartDate',
        'Deleted', 'DeletedReason', 'FastF1Generated', 'isAccurate', 'Status', 'Date', 'SessionTime',
        'RelativeDistance', 'Source', 'Compound_y', 'TyreLife_y', 'TimeXY'
    ]

    # Drop irrelevant columns if they exist in the dataset
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Check existing columns in the dataset
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    non_normalized_cols = [col for col in non_normalized_cols if col in df.columns]

    # Handle pit stop time as a derived feature
    if 'PitOutTime' in df.columns and 'PitInTime' in df.columns:
        df['PitStopTime'] = (df['PitOutTime'] - df['PitInTime']).fillna(0).astype(float)
        df = df.drop(columns=['PitOutTime', 'PitInTime'], errors='ignore')
        numerical_cols.append('PitStopTime')

    # Handle DistanceToDriverAhead
    if 'Position' in df.columns:
        df.loc[df['Position'] == 1, 'DistanceToDriverAhead'] = 0
        df.loc[df['Position'] == 1, 'DriverAhead'] = 0

    # if DriverAhead is null, drop row
    df = df.dropna(subset=['DriverAhead'])

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # Standardize numerical features
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),  # Encode categorical features
        ],
        remainder='passthrough'  # Retain non-normalized features as-is
    )

    print("\n", df.dtypes, "\n")

    # Ensure all boolean columns are converted to int (otherwise it cause a problem)
    df = df.apply(lambda col: col.map({True: 1, False: 0}) if col.dtypes == 'bool' else col)

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"Preprocessing took {elapsed_time:.2f} minutes.")
    return processed_data


# Updated function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Loads FastF1 data and preprocesses it for sequence modeling.
    Args:
        file_path (str): Path to the dataset file (CSV or similar).
    Returns:
        array: Preprocessed dataset ready for sequence modeling.
    """
    dtype_dict = {
            "DriverNumber": int,
            "LapNumber": int,
            "Stint": int,
            "SpeedI1": float,
            "SpeedI2": float,
            "SpeedFL": float,
            "SpeedST": float,
            "IsPersonalBest": bool,
            "Compound_x": str,
            "Compound_y": str,
            "TyreLife_x": int,
            "TyreLife_y": int,
            "FreshTyre": bool,
            "Team": str,
            "TrackStatus": int,
            "Position": int,
            "Deleted": bool,
            "DeletedReason": str,
            "FastF1Generated": bool,
            "IsAccurate": bool,
            "AirTemp": float,
            "Humidity": float,
            "Pressure": float,
            "Rainfall": bool,
            "TrackTemp": float,
            "WindDirection": float,
            "WindSpeed": float,
            "DistanceToDriverAhead": float,
            "RPM": int,
            "Speed": int,
            "nGear": int,
            "Throttle": int,
            "Brake": bool,
            "DRS": int,
            "Source": str,
            "Distance": float,
            "RelativeDistance": float,
            "Status": str,
            "X": int,
            "Y": int,
            "Z": int,
            "Year": int,
            "Event": str,
            # Specify time columns as object
            "PitInTime": object,
            "PitOutTime": object,
            "Sector1SessionTime": object,
            "Sector1Time": object,
            "Sector2SessionTime": object,
            "Sector2Time": object,
            "Sector3SessionTime": object,
            "Sector3Time": object,
            "LapStartTime": object,
            "SessionTime": object,
            "LapTime": object,
            "TimeXY": object,
            "LapStartDate": object,
            "Date": object,
    }

    # Load dataset
    print(f'Loading dataset from {file_path}')
    start_time = time.time()
    np_data = np.load(file_path, allow_pickle=True)
    print(f'Loaded! Converting to dataframe...')
    data = pd.DataFrame(np_data['data'], columns=['Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime',
                                                  'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
                                                  'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
                                                  'Compound_x', 'TyreLife_x', 'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate',
                                                  'TrackStatus', 'Position', 'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate',
                                                  'Compound_y', 'TyreLife_y', 'TimeXY', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall',
                                                  'TrackTemp', 'WindDirection', 'WindSpeed', 'Date', 'SessionTime', 'DriverAhead',
                                                  'DistanceToDriverAhead', 'Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS',
                                                  'Source', 'Distance', 'RelativeDistance', 'Status', 'X', 'Y', 'Z', 'Year', 'Event'])
    data = data.astype(dtype_dict)
    print('Dataset loaded and converted to dataframe')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Dataset loaded in: {elapsed_time:.2f} seconds')

    
    # Preprocess data
    processed_data = normalize_data(data)
    
    return processed_data


def plot_training_history(history):
    """
    Plots training and validation loss over epochs.
    
    Args:
        history (dict): Training history containing 'loss' and 'val_loss'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if any(val is not None for val in history['val_loss']):
        val_loss = [val if val is not None else np.nan for val in history['val_loss']]
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_reconstruction_error(autoencoder, X_val):
    """
    Calculates reconstruction error for validation data.
    
    Args:
        autoencoder (Model): The trained autoencoder.
        X_val (ndarray): Validation data (3D array with shape [samples, sequence_length, features]).
    
    Returns:
        ndarray: Reconstruction error for each sample in the validation data.
    """
    autoencoder.eval()
    with torch.no_grad():
        X_val = torch.tensor(X_val, dtype=torch.float32).to(next(autoencoder.parameters()).device)
        reconstructed_data = autoencoder(X_val).cpu().numpy()
    reconstruction_error = np.mean((X_val.cpu().numpy() - reconstructed_data)**2, axis=(1, 2))
    return reconstruction_error


def compare_reconstruction(autoencoder, X_val, num_samples):
    """
    Compares original data with reconstructed data for selected samples.
    
    Args:
        autoencoder (Model): The trained autoencoder.
        X_val (ndarray): Validation data.
        num_samples (int): Number of validation samples to check reconstruction.
    
    Outputs:
        - Prints original vs reconstructed data for the selected number of samples.
        - Displays reconstruction error for each sample.
    """
    # Select the first `num_samples` from X_val (ensure X_val has sufficient samples)
    sample_data = X_val[:num_samples]

    # Get the reconstructed data
    autoencoder.eval()
    with torch.no_grad():
        sample_data_tensor = torch.tensor(sample_data, dtype=torch.float32).to(next(autoencoder.parameters()).device)
        reconstructed_data = autoencoder(sample_data_tensor).cpu().numpy()

    # Loop over the samples and print a compact comparison
    for i in range(len(sample_data)):  # Use the actual number of samples in sample_data
        print(f"\nSample {i+1} Reconstruction:")
        print(f"- Reconstruction Error: {np.mean((sample_data[i] - reconstructed_data[i])**2):.4f}")

        # Print a small part of the original and reconstructed data to avoid overwhelming output
        print(f"- Original (sample values): {sample_data[i, 0, :5]}...")  # First 5 features (you can adjust this)
        print(f"- Reconstructed (sample values): {reconstructed_data[i, 0, :5]}...")
        print("-" * 50)


def training_diagnostics(autoencoder, history, X_val, sequence_length, num_samples=5):
    """
    Performs diagnostics after training an autoencoder, including:
    - Plotting training and validation loss.
    - Printing final training and validation loss.
    - Calculating mean reconstruction error on validation data.
    - Comparing original and reconstructed sequences.

    Args:
        autoencoder (Model): The trained autoencoder.
        history (dict): Training history containing 'loss' and 'val_loss'.
        X_val (ndarray): Validation data.
        sequence_length (int): Length of the input sequences.
        num_samples (int): Number of samples to display for reconstruction comparison.
    """
    # Ensure X_val is in the correct shape for reconstruction comparison
    if len(X_val.shape) == 2:  # If shape is (num_samples, num_features)
        X_val = np.expand_dims(X_val[:sequence_length], axis=0)  # Add batch and temporal dimensions

    print("\n--- Training Diagnostics ---\n")
    
    # Plot training and validation loss
    plot_training_history(history)
    
    # Final training and validation loss
    final_train_loss = history['loss'][-1]
    final_val_loss = next((val for val in reversed(history['val_loss']) if val is not None), None)
    print(f"Final Training Loss: {final_train_loss:.4f}")
    if final_val_loss is not None:
        print(f"Final Validation Loss: {final_val_loss:.4f}")
    else:
        print("Validation Loss: Not available (validation skipped for some epochs).")
    
    # Calculate mean reconstruction error
    print("\nReconstruction Error Analysis:")
    reconstruction_error = calculate_reconstruction_error(autoencoder, X_val)
    print(f"Mean Reconstruction Error (Validation Data): {np.mean(reconstruction_error):.4f}")
    
    # Display reconstruction comparison for selected samples
    print("\n--- Reconstruction Comparison ---\n")
    compare_reconstruction(autoencoder, X_val, num_samples)


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


def train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    history = {"loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            

        train_loss /= len(train_loader)
        history["loss"].append(train_loss)

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
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            history["val_loss"].append(None)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Validation skipped.")

    return history


# Main function
if __name__ == "__main__":

    """
    Main execution function to train the LSTM autoencoder on FastF1 data
    """

    # clear the output
    os.system('cls' if os.name == 'nt' else 'clear')

    # Check for CUDA/GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available. Training will use the GPU.")
    else:
        print("CUDA is not available. Training will use the CPU.")
    
    # Path to the dataset
    dataset_path = "AllTelemetryData/2023_all_data.npz"
    
    # Load and preprocess data
    data = load_and_preprocess_data(dataset_path)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    num_features = data.shape[1]
    
    # Parametri
    sequence_length = 10
    batch_size = 128
    epochs = 10
    validation_freq = 5

    # Build the autoencoder
    autoencoder = LSTMAutoencoder(sequence_length, num_features).to(device)
    print(autoencoder)

    # Create datasets and loaders
    train_dataset = FastF1Dataset(train_data, sequence_length)
    val_dataset = FastF1Dataset(val_data, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the autoencoder
    history = train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, device)

    # Call diagnostics
    training_diagnostics(autoencoder, history, val_data, sequence_length)
    
    # Save the trained model
    print("Saving the trained model...")
    torch.save(autoencoder.state_dict(), "autoencoder_fastf1.pth")
    print("Model saved as 'autoencoder_fastf1.pth'")