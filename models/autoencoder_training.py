import time
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


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
        if file.endswith('.npz') and file.startswith('2019'):
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
  
    # np_data = np.load('anomaly_normalized/2019_AD_normalized.npz', allow_pickle=True)['data']
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Dataset loaded in: {elapsed_time:.2f} seconds')
    
    return np_data


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


def determine_threshold(reconstruction_errors, percentile=95):
    """
    Determines a threshold for anomalies based on reconstruction errors.
    
    Args:
        reconstruction_errors (ndarray): Reconstruction errors of the training data.
        percentile (float): Percentile to use as the threshold (default is 95th).
    
    Returns:
        float: Calculated threshold for anomaly detection.
    """
    return np.percentile(reconstruction_errors, percentile)

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
    # Ensure X_val is reshaped for sequence modeling
    if len(X_val.shape) == 2:  # (num_samples, num_features)
        sequence_data = [
            X_val[i: i + sequence_length] for i in range(len(X_val) - sequence_length + 1)
        ]
        X_val = np.array(sequence_data)

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


def train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, device, learning_rate, criterion):
    scaler = GradScaler('cuda')
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    history = {"loss": [], "val_loss": []}

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


def evaluate_model(autoencoder, X_test, y_test, threshold):
    """
    Evaluates the anomaly detection performance of the autoencoder.
    
    Args:
        autoencoder (Model): Trained autoencoder.
        X_test (ndarray): Test data.
        y_test (ndarray): Ground truth labels (1 for anomaly, 0 for normal).
        threshold (float): Anomaly detection threshold.
    
    Returns:
        dict: Precision, Recall, F1 Score, and AUC.
    """
    reconstruction_error = calculate_reconstruction_error(autoencoder, X_test)
    predictions = (reconstruction_error > threshold).astype(int)
    
    metrics = {
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, reconstruction_error),
    }
    return metrics


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
    dataset_path = "AD_supernormalized"
    
    # Load and preprocess data
    data = load_data(dataset_path)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    num_features = data.shape[1]

    print("Any NaNs in training data:", np.isnan(train_data).any())
    print("Any NaNs in validation data:", np.isnan(val_data).any())
    print("Any Infs in training data:", np.isinf(train_data).any())
    print("Any Infs in validation data:", np.isinf(val_data).any())

    
    # Parametri
    sequence_length = 20
    batch_size = 128
    epochs = 10
    validation_freq = 5
    learning_rate = 0.001
    criterion = nn.L1Loss() # MAE loss

    # Build the autoencoder
    autoencoder = LSTMAutoencoder(sequence_length, num_features).to(device)
    print(autoencoder)

    # Create datasets and loaders
    train_dataset = FastF1Dataset(train_data, sequence_length)
    val_dataset = FastF1Dataset(val_data, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the autoencoder
    history = train_autoencoder(autoencoder, train_loader, val_loader, epochs, validation_freq, device, learning_rate, criterion)

    # Step 4: Calculate reconstruction error for training set
    train_reconstruction_error = calculate_reconstruction_error(autoencoder, train_data)

    # Step 4.1: Determine anomaly threshold based on the training reconstruction errors
    anomaly_threshold = determine_threshold(train_reconstruction_error, percentile=95)
    print(f"Anomaly Threshold (95th Percentile): {anomaly_threshold:.4f}")

    # Step 5: Test and Evaluate
    # Assuming 'X_test' and 'y_test' are your test features and labels
    metrics = evaluate_model(autoencoder, X_test, y_test, anomaly_threshold)
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Call diagnostics
    training_diagnostics(autoencoder, history, val_data, sequence_length)
    
    # Save the trained model
    print("Saving the trained model...")
    torch.save(autoencoder.state_dict(), "autoencoder_fastf1.pth")
    print("Model saved as 'autoencoder_fastf1.pth'")