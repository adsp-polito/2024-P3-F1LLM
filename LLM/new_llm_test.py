import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define the CNN-LSTM model structure for anomaly classification
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
        x = x.permute(0, 2, 1)  # Switch to (batch_size, n_features, sequence_length)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # Switch back to (batch_size, sequence_length, features)
        x, _ = self.lstm(x)

        x = x[:, -1, :]  # Use the last LSTM output
        x = self.fc(x)
        return x


# Dataset wrapper
class SingleDriverDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LLMModel():
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"  # for Kaggle, maximum 3B parameters
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.chat_history = [
            {"role": "system", "content": "You are a F1 telemetrist helper"}
        ]


# Prediction function for anomaly classification
def predict_anomaly(model_path, file_path, sequence_length, n_classes, anomaly_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    sample_data = np.load(file_path)["data"]
    n_features = sample_data.shape[1] - 1
    model_class = CNNLSTMModel(n_features=n_features, n_classes=n_classes)
    model_class.load_state_dict(torch.load(model_path, map_location=device))
    model_class.to(device)
    model_class.eval()

    # Process the data
    driver_data = sample_data[:, :-1]
    sliding_windows = [
        driver_data[i:i + sequence_length] for i in range(len(driver_data) - sequence_length + 1)
    ]
    sliding_windows = np.array(sliding_windows)

    dataset = SingleDriverDataset(sliding_windows)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    probabilities = torch.zeros((len(sliding_windows), n_classes), device=device)
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model_class(inputs)
            probabilities[idx] = F.softmax(outputs, dim=1).squeeze()

    mean_probabilities = probabilities.mean(dim=0).cpu().numpy()
    aggregated_probabilities = {anomaly_classes[i]: float(mean_probabilities[i]) for i in range(n_classes)}

    most_probable_class_idx = mean_probabilities.argmax()
    predicted_class = anomaly_classes[most_probable_class_idx]

    return predicted_class, aggregated_probabilities


# Conversational Model (GPT-2) Setup


# Chatbot response using GPT-2 model
def generate_chatbot_response(input_text, llm_model: LLMModel, classification_results=None):
    # Prepare context
    context = input_text
    if classification_results:
        context += f" Classification Results:\n{classification_results}"

    try:
        # Append user input to chat history but exclude it from the assistant's output
        llm_model.chat_history.append({"role": "user", "content": context})
        chat_history_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in llm_model.chat_history])

        # Encode the chat history for model input
        inputs = llm_model.tokenizer.encode(chat_history_str, return_tensors="pt")

        # Generate the response
        with torch.no_grad():
            outputs = llm_model.model.generate(
                inputs,
                max_length=1000,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                top_k=50,
                top_p=0.9,
                do_sample=True,
            )
        print(outputs)
        response = llm_model.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append the generated response to chat history
        llm_model.chat_history.append({"role": "assistant", "content": response})
        # response = response.replace(classification_results, "")
        response = response.replace(chat_history_str, "")
        return response.strip()
    except Exception as e:
        return f"Error generating response: {e}"


# Updated Gradio Interface Handling
def handle_input(input_text, file, model, tokenizer):
    classification = ""
    if file is not None:
        # When a file is uploaded, process the anomaly classification
        model_path = "/kaggle/input/anomalyclassifier/pytorch/test1/1/classification_model_fold5_epoch20_loss0.0231.pth"
        sequence_length = 20
        anomaly_classes = ["Others", "Braking System", "Engine", "Power Unit", "Cooling System",
                           "Suspension and Drive", "Aerodynamics and Tyres", "Transmission and Gearbox"]
        n_classes = len(anomaly_classes)

        predicted_anomaly, aggregated_probabilities = predict_anomaly(
            model_path=model_path,
            file_path=file.name,
            sequence_length=sequence_length,
            n_classes=n_classes,
            anomaly_classes=anomaly_classes
        )

        # Format output
        sorted_probs = sorted(aggregated_probabilities.items(), key=lambda x: x[1], reverse=True)
        probabilities_output = "\n".join([f"{cls}: {prob:.2%}" for cls, prob in sorted_probs if prob > 0.001])

        classification = f"Predicted Anomaly: {predicted_anomaly}\n\nAggregated Probabilities:\n{probabilities_output}"

    if input_text:
        # Generate a response without showing the input text in the output
        response = generate_chatbot_response(input_text, llm, classification)
        print("This is the response: ", response)
        return response
    else:
        return "Please upload a file or enter a message."


# Initialize conversational model
llm = LLMModel()

# Create Gradio interface
interface = gr.Interface(
    fn=handle_input,
    inputs=[gr.Textbox(label="Enter a message"), gr.File(label="Upload a .npz file")],
    outputs="text",
    title="Anomaly Classification and Conversational Chatbot",
    description="Chat with me about the anomaly classification model or upload an `.npz` file to classify driver anomalies."
)

if __name__ == "__main__":
    interface.launch()