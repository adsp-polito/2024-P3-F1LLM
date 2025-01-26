# F1LLM: Intelligent Agent for Formula 1 Telemetry Analysis

![Python](https://img.shields.io/badge/python-3.9-blue) 
![NumPy](https://img.shields.io/badge/numpy-1.21.0-blue)
![Pandas](https://img.shields.io/badge/pandas-1.3.3-green)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-orange)
![PyTorch](https://img.shields.io/badge/pytorch-1.12-orange) 
![Gradio](https://img.shields.io/badge/gradio-3.0-red)
![Hugging Face](https://img.shields.io/badge/huggingface-transformers-yellow) 
![Status](https://img.shields.io/badge/status-Completed-brightgreen)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

<img src="assets/logo.png" alt="Project Logo" align="right" width="75" style="margin-left:10px;" />
<img src="assets/dbmg.png" alt="DBMG Logo" align="right" width="75" style="margin-left:10px;" />

The complete report detailing our methodology and results is available 

The comprehensive project report, which provides an in-depth explanation of our methodology, experimental setup, and detailed results, is available for review **[here](https://github.com/adsp-polito/2024-P3-F1LLM/blob/main/Team6_P3_F1LLM_report.pdf)**.

## Overview
In Formula 1, data plays a crucial role in understanding vehicle performance, optimizing strategies, and diagnosing issues. Teams rely on vast amounts of telemetry data collected during races to monitor critical parameters such as speed, engine performance, and tire conditions. However, analyzing this data to identify failures or anomalies is a time-consuming and complex task that requires a deep understanding of the car’s systems and racing environment.

F1LLM addresses this challenge by providing an intelligent system designed to assist telemetrists in post-race failure analysis. The system combines advanced machine learning models—an LSTM-based autoencoder for detecting anomalies in telemetry patterns and a CNN-LSTM-FC classifier for categorizing failures into specific types—with the power of a Large Language Model (LLM) for interpreting and explaining results. Together, these components process telemetry signals from the FastF1 dataset (2019–2024), enabling the identification of irregularities and the classification of failures with greater efficiency and accuracy.

By streamlining the analysis process, F1LLM not only saves time but also provides actionable insights that can help teams address recurring issues and improve future performance. This makes it a valuable tool in the high-stakes environment of Formula 1, where quick and accurate decision-making is essential.

---

## Dataset

### Source
- **FastF1**: This open-source library provides a comprehensive collection of telemetry signals, weather data, and race-related variables. It includes high-resolution multivariate time-series data, essential for detecting anomalies and diagnosing failures in Formula 1 vehicles.
- **Scope**: The dataset focuses on races held between 2019 and 2024, covering a wide range of environmental and performance conditions to ensure robustness in the analysis.

### Preprocessing
Preprocessing was a critical step in ensuring the quality and usability of the telemetry data for machine learning models in this project. This process involved several key tasks:

1. **Understanding the Data**:
   - Significant effort was spent analyzing the dataset to understand which telemetry signals were relevant for anomaly detection and failure classification.
   - Domain knowledge and iterative feature analysis were employed to identify the most impactful features, such as RPM, Speed, Throttle, and Weather Conditions.

2. **Data Cleaning**:
   - Rows with missing values (NaN) were removed to maintain the integrity of the data.
   - Duplicate records were identified and eliminated to avoid bias during model training.
   - Anomalous values, such as extreme outliers not associated with real-world racing conditions, were filtered out.
   - A significant effort was dedicated to correcting inconsistencies and inaccuracies in the FastF1 dataset. As the library compiles telemetry data from public sources, some records were mislabeled or incomplete, particularly for failures leading to driver retirements. These issues were carefully addressed to ensure the data accurately reflected real-world scenarios and could be reliably used for machine learning tasks.

3. **Feature Selection**:
   - The initial set of 61 telemetry signals was systematically reduced to 27 critical variables through correlation analysis and expert review.
   - Variables like specific track segment speeds (e.g., SpeedFL, SpeedI1) and less informative features such as Stint and Sector Time were excluded.
   - Selected features included both car-related metrics (e.g., RPM, Throttle) and environmental data (e.g., Humidity, Wind Speed), ensuring a balanced representation of factors influencing race performance.

4. **Normalization**:
   - Numerical features were scaled using Min-Max normalization to ensure uniformity and improve model stability during training.
   - Time-series data were normalized across each feature to highlight relative changes without distorting absolute patterns.

5. **Categorical Encoding**:
   - Non-numerical variables, such as compound types, were encoded into numerical formats where necessary to enable their use in machine learning models.

6. **Data Partitioning**:
   - The telemetry data was divided into:
     - **Training and Validation Data**: Telemetry from 2019 to 2023, excluding laps with failures, to train models on normal racing patterns.
     - **Testing Data**: Data from the 2024 season, including anomalies, to evaluate the system’s ability to generalize to new and unseen failures.

7. **Storage and Format**:
   - To optimize computational efficiency, the cleaned and processed datasets were stored in NPZ format, which allows for faster data loading and streamlined integration with Python workflows.

By thoroughly preprocessing the dataset, we ensured that the data fed into the models was not only clean and reliable but also representative of the conditions encountered in real-world Formula 1 scenarios. This foundational work significantly enhanced the accuracy and robustness of the anomaly detection and failure classification systems.

---

## Methodology
### System Components
1. **Anomaly Detection**:
   - *Model*: LSTM Autoencoder.
   - *Function*: Identifies irregularities in telemetry by reconstructing normal data patterns.
   - *Threshold*: Based on the 99th percentile of reconstruction errors.

2. **Failure Classification**:
   - *Model*: CNN-LSTM-FC.
   - *Function*: Assigns probabilities to eight predefined failure categories.
   - *Categories*: Engine, Braking System, Cooling System, Transmission, etc.

3. **LLM Integration**:
   - *Model*: LLaMA 3.2-1B via Hugging Face Transformers.
   - *Function*: Dynamically invokes ML modules based on prompts, synthesizing insights into detailed natural language reports.

---

## Repository Structure
```
2024-P3-F1LLM/
|— anomaly_detection/
    |— autoencoder_training.py
    |— classification_data.py
    |— classification_training.py
    |— threshold.py
|— Checkpoints/
    |— 2024_11_14 First Checkpoint
    |— 2024_12_04 Second Checkpoint
    |— 2025_01_09 Third Checkpoint
    |— 2025_01_31 Final Checkpoint
|— data_extraction_and_preprocessing/
    |— all_failures_retrieving.py
    |— all_race_data_extraction.py
    |— normalize_new_version.py
    |— normalize_npz.py
    |— race_data_extraction.py
|— Dataset/
    |— FailuresTelemetryData/
    |— OnlyFailuresByDriver/
    |— 19-24_all_events_anomalies.csv
    |— Failures_grouped_2018_2024.csv
    |- Failures2014_2024_cleaned.csv
    |- Failures2014_2024
    |- new_pilots.csv
|— LLM/
    |— LLM_launch.py
|— notebooks/
    |— csvAnomalies.ipynb
    |— modelAD_evaluation.ipynb
    |- race_data_extraction.ipynb
    |- testClassificationEvaluation.ipynb
```

#### Description of the Repository Content

- **`anomaly_detection/`**  
  Contains scripts and resources for training machine learning models used for anomaly detection and failure classification.

- **`Checkpoints/`**  
  Contains the complete history of project presentations, from initial drafts to final versions, documenting the evolution of ideas and progress throughout the work period.

- **`data_extraction_and_preprocessing/`**  
  Includes scripts for extracting raw telemetry data from the FastF1 library and preprocessing it to prepare datasets for model training.

- **`Dataset/`**  
  Contains preprocessed datasets, structured data, and organized telemetry information necessary for the analysis.

- **`LLM/`**  
  Houses the implementation of the Large Language Model (LLM) used to interpret telemetry data and generate detailed explanations.

- **`notebooks/`**  
  Includes Jupyter notebooks for exploratory data analysis, visualization, and evaluation of the models and telemetry insights.

---

## Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/2024-P3-F1LLM.git
   cd 2024-P3-F1LLM
   ```
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib gradio torch transformers
   ```

### Running the System
To use the system, follow these steps:

1. **Launch the LLM**:
   - Run the `LLM_launch.py` file:
     ```bash
     python LLM/LLM_launch.py
     ```
   - Insert your Hugging Face API token when prompted in the script.
   - Choose the desired models for anomaly detection and failure classification by editing the corresponding variables in the code.
   - Copy the link printed in the terminal and open it in your browser to access the Gradio-based chatbot interface.

2. **Using the Chatbot**:
   - On the Gradio interface, follow the instructions provided to interact with the system.
   - Ask the chatbot to perform either **anomaly detection** or **failure classification**, and upload the required telemetry data when prompted.
   - The system will process your input and provide outputs, including anomaly detection results or failure classification probabilities.

### Model Training
If you wish to train the models from scratch, you can use the following scripts:
- **Anomaly Detection**:
  ```bash
  python anomaly_detection/autoencoder_training.py
  ```
- **Failure Classification**:
  ```bash
  python anomaly_detection/classification_training.py
  ```

Some pre-trained models are included in the repository, in their respective folders.

---

## Results

### Failure Classification Performance
The table below summarizes the performance of the CNN-LSTM-FC classification model on telemetry data from the 2024 season. It highlights the actual and predicted failure categories for key events, along with the percentage confidence of the Top-1 prediction for the predicted failure category.

| Event              | Driver | Actual Failure       | Predicted Failure        | Confidence (%) |
|--------------------|--------|----------------------|--------------------------|----------------|
| Hungarian GP       | GAS    | Suspension and Drive | Suspension and Drive     | 78.00          |
| Las Vegas GP       | GAS    | Engine               | Engine                   | 81.05          |
| Saudi Arabian GP   | GAS    | Transmission and Gearbox | Transmission and Gearbox | 81.98          |
| Mexico City GP     | ALO    | Braking System       | Transmission and Gearbox | 78.87          |
| Canadian GP        | LEC    | Engine               | Engine                   | 91.58          |
| Australian GP      | VER    | Braking System       | Others                   | 81.08          |
| Italian GP         | TSU    | Cooling System       | Transmission and Gearbox | 28.75          |
| Las Vegas GP       | ALB    | Cooling System       | Transmission and Gearbox | 51.73          |
| Singapore GP       | ALB    | Power Unit           | Cooling System           | 52.46          |
| Japanese GP        | ZHU    | Transmission and Gearbox | Transmission and Gearbox | 45.51          |
| Australian GP      | HAM    | Engine               | Others                   | 65.86          |
| British GP         | RUS    | Cooling System       | Cooling System           | 90.09          |

The confidence percentage represents the likelihood assigned by the model to its Top-1 predicted failure category. While the model performs well for categories like **Engine** and **Transmission and Gearbox**, it struggles with others like **Braking System**, where failures are binary (e.g., True/False) and underrepresented in the dataset. This limitation impacts the model's ability to learn complex patterns for such failure types, highlighting the need for future improvements.

### Key Observations
- The **LSTM Autoencoder** demonstrated strong performance in anomaly detection, achieving a reconstruction loss of 0.4. Anomalies detected aligned closely with known telemetry irregularities.
- The **CNN-LSTM-FC Classifier** achieved 50% accuracy for Top-1 predictions and 75% for Top-3 predictions, highlighting its capability to provide meaningful failure categorization in most cases.
- Certain failure categories, such as **Braking System**, proved difficult to classify accurately due to their binary structure (e.g., True/False) and limited representation in the dataset.
- The dataset’s imbalance and scarcity of failure-specific telemetry data affected the classifier’s ability to generalize to underrepresented categories, suggesting the need for synthetic data generation or expanded datasets in future iterations.

---

## Future Implementations
- **Enhanced Datasets**: Expanding the dataset to include more diverse failure scenarios and additional telemetry variables to improve model generalization.
- **Improved Models**: Refining the classification model to address overlapping categories and enhance accuracy in complex scenarios.
- **Advanced LLMs**: Leveraging more powerful language models to handle complex queries and provide deeper insights.
- **Additional Modules**: Implementing lap prediction and telemetry comparison functionalities to further support telemetrists.

---

## Conclusion

This project demonstrated the integration of machine learning models and an LLM to analyze Formula 1 telemetry data. The system streamlined the failure analysis process by detecting anomalies, classifying failures, and providing actionable insights. Despite its achievements, the project highlighted critical areas for improvement, particularly in dataset diversity and model generalization.

Future work should focus on addressing these limitations by expanding the dataset, generating synthetic data to balance failure types, and fine-tuning models to handle edge cases. Additionally, leveraging more advanced LLMs and incorporating supplementary modules like lap prediction and telemetry comparison could further enhance the system's capabilities.

By building on the foundation established in this project, the proposed system has the potential to become a robust tool for real-time telemetry analysis, aiding Formula 1 teams in optimizing performance and making informed decisions in high-pressure scenarios.

---

## Contributors
- **[Davide Benotto](https://github.com/DavideBenotto01)**  
- **[Manuele Mustari](https://github.com/Manuele23)**  
- **[Paolo Riotino](https://github.com/paoloriotino)** 

---

## License
This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

## Acknowledgments

- [Politecnico di Torino](https://www.polito.it)
- [DataBase and Data Mining Group - Politecnico di Torino](https://dbdmg.polito.it/dbdmg_web/)