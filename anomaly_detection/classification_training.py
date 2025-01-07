import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

new_data_path = "../Dataset/OnlyFailuresByDriver/npz_failures_MinMaxScaler_normalized.npz"

# Load the new data
new_data = np.load(new_data_path, allow_pickle=True)['data']
new_data_array = np.array(new_data, dtype=np.float32)

# Split the data into features and target
X = new_data_array[:, :-1]  # All columns except the last one
y = new_data_array[:, -1]   # The last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)