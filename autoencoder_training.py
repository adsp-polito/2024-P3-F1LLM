import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model   # type: ignore
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense   # type: ignore
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
    
    # List of time columns to convert
    time_columns = [
        'Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'PitOutTime', 'PitInTime'
    ]

    # Convert all time columns to seconds
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_time_to_seconds)

    # Columns to normalize
    numerical_cols = [
        'Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'Speedl1', 'Speedl2', 'SpeedFL', 'Speed', 'AirTemp', 'Humidity',
        'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed',
        'DistanceToDriverAhead', 'RPM', 'nGear', 'Throttle', 'Brake',
        'DRS', 'X', 'Y', 'Z', 'Distance'
    ]

    # THIS IS TEMPORARY: WE NEED TO MANAGE BETTER THE "OBJECT" TYPE

    # Handle `Compound` with OneHotEncoder
    if 'Compound' in df.columns:
        one_hot = pd.get_dummies(df['Compound'], prefix='Compound')
        df = pd.concat([df, one_hot], axis=1).drop(columns=['Compound'])

    # Map Event column to numeric identifiers
    if 'Event' in df.columns:
        df, event_mapping = map_grand_prix(df)
        # print("Mapped Grand Prix events:", event_mapping)

    # Columns to one-hot encode
    categorical_cols = ['TrackStatus']

    # Columns to leave unprocessed
    non_normalized_cols = ['DriverNumber', 'Stint', 'LapNumber', 'Position', 'IsPersonalBest', 'Year', 'Event']

    # Columns to drop (irrelevant or redundant)
    drop_cols = [
        'Driver', 'SpeedST', 'TyreLife', 'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate',
        'Deleted', 'DeletedReason', 'FastF1Generated', 'isAccurate', 'Status', 'Date', 'SessionTime',
        'RelativeDistance', 'Source', 'DriverAhead', 
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
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Preprocess data
    processed_data = normalize_data(data)
    
    return processed_data


import matplotlib.pyplot as plt
import numpy as np

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
    reconstructed_data = autoencoder.predict(X_val, verbose=0)
    reconstruction_error = np.mean((X_val - reconstructed_data)**2, axis=(1, 2))
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
    reconstructed_data = autoencoder.predict(sample_data)

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


def sliding_window_generator(data, sequence_length, batch_size):
    """
    Generates sliding windows dynamically for training and validation
    Args:
        data (array): Input data.
        sequence_length (int): Number of timesteps in each sequence.
        batch_size (int): Number of sequences per batch.
    Yields:
        tuple: Batch of sequences (X, X) for autoencoder training.
    """
    num_samples = len(data) - sequence_length + 1
    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = np.array([data[i:i+sequence_length] for i in range(start_idx, end_idx)])
            # When the yield statement is executed, the generator state is frozen and the value of the expression_list 
            # is returned to the next() call
            yield batch_data, batch_data    
            # the input and the label (X and y) are identical, typical for an autoencoder, since the model has to reconstruct the input data

def train_autoencoder(
    autoencoder,
    train_gen,
    val_gen,
    steps_per_epoch,
    validation_steps,
    epochs,
    validation_freq,
    other_metrics=["mae"],
):
    """
    Trains the autoencoder using Keras' fit method with custom metrics and validation frequency.

    Args:
        autoencoder (Model): The autoencoder model.
        train_gen (generator): Generator for training data.
        val_gen (generator): Generator for validation data.
        steps_per_epoch (int): Number of steps (batches) per epoch for training.
        validation_steps (int): Number of steps (batches) per epoch for validation.
        epochs (int): Number of epochs to train.
        validation_freq (int): Frequency (in epochs) for validation.
        metrics (list): List of metrics to track during training.
    
    Returns:
        History: Training history object returned by Keras fit.
    """

    # Compile the model with custom metrics
    autoencoder.compile(optimizer="adam", loss="mse", metrics=other_metrics)

    history = {"loss": [], "val_loss": []}
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Determine if validation should be used this epoch
        validate_this_epoch = (epoch) % validation_freq == 0 or (epoch) == epochs or (epoch) == 1
        
        # Train for the current epoch
        epoch_history = autoencoder.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen if validate_this_epoch else None,
            validation_steps=validation_steps if validate_this_epoch else 0,
            verbose=1,
        )

        # Store training loss
        history["loss"].append(epoch_history.history["loss"][-1])
        
        # Store metrics (only the ones you want to print like 'mae')
        for metric in other_metrics:
            history[metric] = history.get(metric, [])  # Ensure key exists in history
            history[metric].append(epoch_history.history.get(metric, [None])[-1])  # Safe retrieval

        # Store validation loss and metrics if validation was performed
        if "val_loss" in epoch_history.history:
            history["val_loss"].append(epoch_history.history["val_loss"][-1])
            print(f"Epoch ({epoch}): Validation MSE -> {history['val_loss'][-1]:.4f}")
            
            # Print validation metrics (only the ones in 'metrics' list)
            for metric in other_metrics:
                val_metric = f"val_{metric}"
                if val_metric in epoch_history.history:
                    print(f"Epoch ({epoch}): Validation {metric.upper()} -> {epoch_history.history[val_metric][-1]:.4f}")
        else:
            history["val_loss"].append(None)
            print(f"Epoch {epoch}: Validation skipped.")
    
    return history


# Function to create the LSTM autoencoder model
def build_autoencoder(sequence_length, num_features, other_metrics):

    """
    Builds an LSTM autoencoder for sequence data.
    Args:
        sequence_length (int): Number of timesteps in each sequence.
        num_features (int): Number of features per timestep.
    Returns:
        Model: Compiled LSTM autoencoder model.

    LSTM Autoencoder Architecture

    The model is designed to process sequential data (e.g., time-series or telemetry data) using an encoder-decoder structure:

    Encoder:
    - Extracts temporal dependencies in the input sequence.
    - Gradually compresses the input into a single latent vector, which summarizes the entire sequence.

    Latent Space:
    - Acts as a bottleneck, forcing the model to focus on essential patterns in the data.
    - Discards noise and redundant information.

    Decoder:
    - Reconstructs the original sequence step-by-step from the latent vector.
    - Expands the latent representation back to the original sequence shape.

    Objective:
    - The model is trained to minimize reconstruction error (Mean Squared Error, MSE) between the input sequence and the reconstructed sequence.
    - High reconstruction error on unseen data indicates anomalies or deviations from learned patterns.
    """

    # Input layer
    # Accepts a sequential input of shape (timesteps, features)
    input_layer = Input(shape=(sequence_length, num_features))
    
    # Encoder
    # Extracts temporal dependencies and outputs intermediate representations
    encoded = LSTM(64, activation='relu', return_sequences=True)(input_layer)
    # Compresses the sequence into a single latent vector summarizing the input
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)

    # Latent Space
    # Repeats the latent vector for each timestep to initialize the decoder
    latent_space = RepeatVector(sequence_length)(encoded)

    # Decoder
    # Begins reconstructing the sequence step-by-step
    decoded = LSTM(32, activation='relu', return_sequences=True)(latent_space)
    # Expands the sequence to match the original complexity
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    
    # Reconstructs the sequence to have the same features as the input
    output_layer = TimeDistributed(Dense(num_features))(decoded)
    
    # Define the autoencoder
    # Optimizes the model to minimize reconstruction error (MSE)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse", metrics = other_metrics)

    return autoencoder


# Main function
if __name__ == "__main__":

    """
    Main execution function to train the LSTM autoencoder on FastF1 data
    """

    # clear the output
    os.system('cls' if os.name == 'nt' else 'clear')

    # Check for CUDA/GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("CUDA is available. Training will use the GPU.")
        print("Available GPU(s):", tf.config.list_physical_devices('GPU'))
    else:
        print("CUDA is not available. Training will use the CPU.")
    
    # Path to the dataset
    dataset_path = "/AllTelemetryData/2022/all_drivers_AustralianGrandPrix_2022.csv"
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(dataset_path)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    num_features = data.shape[1]
    
    # Parametri
    sequence_length = 10
    batch_size = 32
    epochs = 1
    validation_freq = 5
    other_metrics = ["mae"]         # MSE is the standard loss => there is no the need to specify it

    # Build the autoencoder
    autoencoder = build_autoencoder(sequence_length, num_features, other_metrics)
    autoencoder.summary()

    # Create generators
    train_gen = sliding_window_generator(train_data, sequence_length, batch_size)
    val_gen = sliding_window_generator(val_data, sequence_length, batch_size)

    # Train the autoencoder
    steps_per_epoch = (len(train_data) - sequence_length + 1) // batch_size
    validation_steps = (len(val_data) - sequence_length + 1) // batch_size

    history = train_autoencoder(
        autoencoder,
        train_gen,
        val_gen,
        steps_per_epoch,
        validation_steps,
        epochs,
        validation_freq,
        other_metrics=other_metrics,  # Personalized Metrics
    )

    # Call diagnostics
    training_diagnostics(autoencoder, history, val_data, sequence_length)
    
    # Save the trained model
    print("Saving the trained model...")
    autoencoder.save("autoencoder_fastf1.keras")
    print("Model saved as 'autoencoder_fastf1.keras'")