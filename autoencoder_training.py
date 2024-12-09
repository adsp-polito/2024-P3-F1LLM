import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to preprocess numerical and categorical data
def normalize_data(df):
    """
    Preprocesses FastF1 data by detecting column types and applying appropriate transformations.
    Args:
        df (DataFrame): Input dataset.
    Returns:
        array: Combined and preprocessed dataset as NumPy array.
    """
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    irrelevant_cols = [col for col in df.columns if col not in numerical_cols and col not in categorical_cols]

    # Drop irrelevant columns
    df = df.drop(columns=irrelevant_cols)

    # The ColumnTransformer applies different preprocessing steps to different types of columns:
    # - 'num': Numerical columns are normalized using StandardScaler, 
    #          which standardizes features by removing the mean and scaling to unit variance.
    # - 'cat': Categorical columns are transformed using OneHotEncoder, 
    #          which converts categories into binary vectors (one-hot encoding).
    # - remainder='drop': Drops any columns not specified in 'numerical_cols' or 'categorical_cols'.
    # This ensures that the numerical and categorical data are preprocessed correctly and combined
    # into a single dataset ready for machine learning models.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # Normalize numerical columns
            ('cat', OneHotEncoder(sparse_output=False), categorical_cols)  # One-hot encode categorical columns
        ],
        remainder='drop'
    )

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)

    return processed_data

# Function to load and preprocess the data
def load_and_preprocess_data(file_path, sequence_length=10):

    """
    Loads FastF1 data, preprocesses the features, and splits it into train and validation sets.
    Args:
        file_path (str): Path to the dataset file (CSV or similar).
        sequence_length (int): Number of timesteps for each input sequence.
    Returns:
        tuple: Preprocessed train and validation datasets (X_train, X_val).
    """
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Preprocess data
    processed_data = normalize_data(data)

    """
    Explanation of sequence creation:
    - Each sequence contains 'sequence_length' timesteps of feature data.
    - The loop starts from the first row (i=0) and creates windows of size
      'sequence_length' that slide over the dataset.
    - For example, if the dataset has 100 rows and sequence_length = 10,
      the first sequence contains rows [0:10], the second contains [1:11],
      and so on, resulting in (100 - 10 + 1) = 91 sequences.
    - The resulting 'sequences' array has dimensions:
      (number of sequences, sequence_length, number of features).
    """
    
    # Create sequences for LSTM
    sequences = []
    for i in range(len(processed_data) - sequence_length + 1):
        sequences.append(processed_data[i : i + sequence_length])
    sequences = np.array(sequences)
    

    
    return sequences

# Function to create the LSTM autoencoder model
def build_autoencoder(sequence_length, num_features):

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
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder



# Function to train the autoencoder with tqdm progress bar
def train_autoencoder(autoencoder, X_train, X_val, epochs=50, batch_size=32):

    """
    Trains the autoencoder with a progress bar and logs metrics during training.
    Args:
        autoencoder (Model): The autoencoder model.
        X_train (array): Training data.
        X_val (array): Validation data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    Returns:
        History: Training history of the autoencoder.
    """

    history = []
    for epoch in tqdm(range(1, epochs + 1), desc="Training Progress"):

        # Train for one epoch
        train_loss = autoencoder.train_on_batch(X_train, X_train)
        
        # Evaluate on validation data
        val_loss = autoencoder.evaluate(X_val, X_val, verbose=0)
        
        # Append metrics to history
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        
        # Print metrics
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    return history

# Main function
if __name__ == "__main__":
    
    """
    Main execution function to train the LSTM autoencoder on FastF1 data.
    """

    # Check for CUDA/GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("CUDA is available. Training will use the GPU.")
        print("Available GPU(s):", tf.config.list_physical_devices('GPU'))
    else:
        print("CUDA is not available. Training will use the CPU.")


    # Path to the dataset (update with your file path)
    dataset_path = "/Users/manuelemustari/Desktop/Università/Politecnico di Torino/2° year/1° period/Applied Data Science Project/Project/2024-P3-F1LLM/Dataset/AllTelemetryData/2022/all_drivers_AustralianGrandPrix_2022.csv"
    
    # Ask the user for parameters
    print("Enter training parameters:")
    epochs = int(input("Number of epochs (Ex. 10, 25, 50, ...): "))
    batch_size = int(input("Batch size (Ex. 8, 16, 32, 64, ...): "))
    sequence_length = int(input("Sequence length (window size for LSTM): "))
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_data = load_and_preprocess_data(dataset_path, sequence_length=sequence_length)
    num_features = X_data.shape[1]
    
    # Split into train and validation sets
    X_train, X_val = train_test_split(X_data, test_size=0.2, random_state=42)
    
    # Build the LSTM autoencoder
    print(f"Building the LSTM autoencoder with sequence length {sequence_length} and {num_features} features...")
    autoencoder = build_autoencoder(sequence_length, num_features)
    autoencoder.summary()
    
    # Train the autoencoder
    print("Training the LSTM autoencoder...")(autoencoder, X_train, X_val, epochs=epochs, batch_size=batch_size)
    
    # Save the trained model
    print("Saving the trained model...")
    autoencoder.save("lstm_autoencoder_model_fastf1.h5")
    print("Model saved as 'lstm_autoencoder_model_fastf1.h5'")