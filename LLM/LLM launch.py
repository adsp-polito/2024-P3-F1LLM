# Install necessary libraries -> Kaggle
# %pip install gradio #type:ignore
# %pip install -U "huggingface_hub[cli]" #type:ignore

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gradio as gr
from transformers import pipeline

# INSERT YOUR HUGGINGFACE TOKEN INSIDE THE FOLLOWING FUNCTION
#from huggingface_hub.hf_api import HfFolder
# HfFolder.save_token()

# Define the CNN-LSTM model structure [FAILURE CLASSIFICATION]
class CNNLSTMModel(nn.Module):
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
        # Forward pass through CNN layers
        x = x.permute(0, 2, 1)  # Reshape input to (batch_size, n_features, sequence_length)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Forward pass through LSTM
        x = x.permute(0, 2, 1)  # Reshape back to (batch_size, sequence_length, features)
        x, _ = self.lstm(x)
        
        # Use the last LSTM output
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Define the LSTM Autoencoder model [ANOMALY DETECTION]
class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features):
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

# Dataset for FastF1 data [ANOMALY DETECTION -> FAILURE CLASSIFICATION]
class FastF1Dataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx + self.sequence_length], dtype=torch.float32)

# Dataset wrapper for a single driver's data [FAILURE CLASSIFICATION]
class SingleDriverDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Anomaly detection function
# This will generate a dataset with detected anomalies for further processing
# SISTEMARE THRESHOLD
def detect_anomalies(autoencoder_model_path, driver_data, sequence_length, threshold=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Remove the last column of driver_data
    driver_data = driver_data[:, :-1]

    # Load the dataset and prepare the DataLoader
    test_dataset = FastF1Dataset(driver_data, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_features = driver_data.shape[1]

    # Load the autoencoder model
    model = LSTMAutoencoder(num_features)
    model.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
    model.to(device)
    model.eval()

    reconstruction_errors = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            batch_errors = torch.mean((inputs - outputs) ** 2, dim=(1)).detach().cpu().numpy()
            reconstruction_errors.extend(batch_errors)
    
    reconstruction_errors = np.array(reconstruction_errors)

    # Determine anomalies based on threshold
    threshold = np.percentile(reconstruction_errors, 99.9)  # 99.9th percentile as threshold
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

    failures_np = driver_data[:, -1]

    # Extract the last block and prepare the dataset
    last_anomaly =  blocks[-1][0] - 200 

    failures_np = failures_np.reshape(-1, 1)
    recombined_array = np.hstack((driver_data, failures_np))

    return recombined_array[last_anomaly:]
    
# Prediction function for anomaly classification
def predict_anomaly(model_path, driver_data, sequence_length, n_classes=8, anomaly_classes=None):

    driver_data = driver_data[:, :-1]  # Remove the last column

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMModel(n_features=driver_data.shape[1], n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Generate sliding windows dynamically
    sliding_windows = []
    for i in range(len(driver_data) - sequence_length + 1):
        window = driver_data[i:i + sequence_length]
        sliding_windows.append(window)
    
    sliding_windows = np.array(sliding_windows)  # Convert to NumPy array

    # Wrap sliding windows in a Dataset and DataLoader
    dataset = SingleDriverDataset(sliding_windows)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Make predictions for each window
    probabilities = torch.zeros((len(sliding_windows), n_classes), device=device)  # Store probabilities for all windows

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities[idx] = F.softmax(outputs, dim=1).squeeze()  # Store probabilities

    # Aggregated Probabilities
    mean_probabilities = probabilities.mean(dim=0).cpu().numpy()  # Average probabilities across all windows

    aggregated_probabilities = {
        anomaly_classes[i]: float(mean_probabilities[i]) for i in range(n_classes)
    }

    # Majority Voting based on probabilities
    most_probable_class_idx = mean_probabilities.argmax()  # Get the class with the highest average probability
    predicted_class = anomaly_classes[most_probable_class_idx]

    return predicted_class, aggregated_probabilities

# Classification function combining anomaly detection and classification
def classify_anomalies(autoencoder_model_path, classification_model_path, driver_data, n_classes, anomaly_classes):
    # Perform anomaly detection

    anomaly_data = detect_anomalies(autoencoder_model_path, driver_data, sequence_length=20)

    if len(anomaly_data) == 0:
        return "No anomalies detected."

    # Perform anomaly classification using the detected anomaly data
    _, aggregated_probabilities = predict_anomaly(
        classification_model_path, 
        anomaly_data, 
        sequence_length=100, 
        n_classes=n_classes, 
        anomaly_classes=anomaly_classes
    )

    # Sort aggregated probabilities by value in descending order
    sorted_probabilities = sorted(aggregated_probabilities.items(), key=lambda x: x[1], reverse=True)


    return sorted_probabilities

class LLM_Agent:
    def __init__(self, autoencoder_model_path, classification_model_path):
        # Paths to the models for anomaly detection and classification
        self.autoencoder_model_path = autoencoder_model_path
        self.classification_model_path = classification_model_path

        # Initialize LLM pipeline
        self.llm = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

        # System context for the LLM
        self.system_context = (
            "#CONTEXT\nROLE: F1-AI assistant.\nTASK: analyze the input data received, "
            "interpret the results, explain their meaning in a clear and concise manner\n"
        )

    def analyze(self, data, task, threshold=None):
        try:
            if task == "anomaly_detection":
                # Detect anomalies using the autoencoder model
                sequence_length = 20
                anomalies = detect_anomalies(self.autoencoder_model_path, data, sequence_length, threshold=threshold)
                return anomalies

            elif task == "classify_anomalies":
                # Predefined anomaly classes
                anomaly_classes = [
                    "Others", "Braking System", "Engine", "Power Unit", "Cooling System", 
                    "Suspension and Drive", "Aerodynamics and Tyres", "Transmission and Gearbox"
                ]
                n_classes = len(anomaly_classes)

                # Classify anomalies using the provided models
                sorted_probabilities = classify_anomalies(
                    self.autoencoder_model_path,
                    self.classification_model_path,
                    data,
                    n_classes,
                    anomaly_classes
                )
                return sorted_probabilities

        except Exception as e:
            # Handle errors gracefully
            return f"An error occurred during analysis: {e}"

    def determine_task(self, prompt):
        if "anomaly detection" in prompt.lower():
            return "anomaly_detection"
        elif "failure classification" in prompt.lower():
            return "classify_anomalies"
        else:
            return "unknown"

    def generate_response(self, instructions, max_length=150):
        try:
            # Concatenate the hidden system context with specific instructions
            full_prompt = self.system_context + "\n" + instructions

            # Generate a response using the LLM
            response = self.llm(
                full_prompt,
                max_length=max_length,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.8,
            )[0]['generated_text']

            return response

        except Exception as e:
            # Handle response generation errors
            return f"Error during response generation: {e}"

    def respond_to_input(self, prompt, data=None, threshold=None):
        try:
            # Determine the task from the user's prompt
            task = self.determine_task(prompt)

            if task == "anomaly_detection":
                if data is None:
                    # If data is missing, request it
                    instructions = (
                        "The anomaly detection task requires telemetry data. "
                        "Please provide the necessary data to proceed (.npz files)."
                    )
                    return instructions

                # Perform anomaly detection
                task_result = self.analyze(data, "anomaly_detection", threshold=threshold)
                instructions = (
                    "Based on the telemetry data, anomalies were detected. "
                    "Anomalies in Formula 1 refer to unexpected or abnormal events during a race session. "
                )
                return self.generate_response(instructions, max_length=300)

            elif task == "classify_anomalies":
                if data is None:
                    # If data is missing, request it
                    instructions = (
                        "The failure classification task requires telemetry data. "
                        "Please provide the necessary data to proceed."
                    )
                    return instructions

                # Perform anomaly classification
                task_result = self.analyze(data, "classify_anomalies")
                formatted_results = "\n".join(
                    [f"- {anomaly}: {probability:.2f} probability" for anomaly, probability in task_result[:3]]
                )
                instructions = (
                    "#RESULTS\n"
                    "Analysis of the telemetry data has been completed. Results:\n"
                    f"{formatted_results}\n"
                    "Now, let's explain what these results mean.\n"
                )
                return self.generate_response(instructions, max_length=500)

            else:
                # Unsupported or unknown task
                instructions = (
                    "I can assist with telemetry analysis tasks such as anomaly detection and failure classification. "
                    "Please specify one of these tasks to proceed."
                )

                return instructions

        except Exception as e:
            # Handle processing errors
            return f"Error during processing: {e}"

# Gradio input handler
def handle_input(input_text, file, agent):
    """
    Handles input text and file for Gradio interface.
    Calls the agent to process either input text, file data, or both.
    """
    if file is not None:
        try:
            # Load the file and extract the data
            data = np.load(file.name, allow_pickle=True)["data"]
        except Exception as e:
            return f"Error loading file: {e}"

        # Call the agent with both input text and data
        response = agent.respond_to_input(input_text, data=data)
        return response
    elif input_text:
        # Call the agent with only input text
        response = agent.respond_to_input(input_text)
        return response
    else:
        return "Please upload a file or provide input."

def main():

    # Initialize the models and agent
    autoencoder_model_path = "/kaggle/input/ad_19-23_autoencoder_adamw_lr0001_ep2_loss0.6264/pytorch/default/1/AD_19-23_autoencoder_AdamW_lr0001_ep2_loss0.6264.pth"
    classification_model_path = "/kaggle/input/classification_model_fold5_epoch20_loss0.0231.pth/pytorch/default/1/classification_model_fold5_epoch20_loss0.0231.pth"

    # Initialize the LLM agent
    agent = LLM_Agent(autoencoder_model_path, classification_model_path)

    # Gradio interface setup
    interface = gr.Interface(
        fn=lambda input_text, file: handle_input(input_text, file, agent),
        inputs=[
            gr.Textbox(label="Enter your message", placeholder="Type something here..."),
            gr.File(label="Upload a .npz file"),
        ],
        outputs="text",
        title="F1LLM - Your F1 Personal Assistant",
        description=(
            "Provide a message to interact with the LLM or upload a `.npz` file "
            "containing telemetry data for anomaly detection and classification. "
            "This tool leverages advanced AI to assist in analyzing race data."
        ),
        allow_flagging="never"
    )

    # Launch the Gradio interface
    interface.launch()

if __name__ == "__main__":

    main()


