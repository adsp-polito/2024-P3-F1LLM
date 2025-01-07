import numpy as np
import pandas as pd
from ..data_extraction_and_preprocessing.normalize_npz import Normalizer
import os

def extract_last_3_laps(npz_file_path, train=True):
    """
    Extracts the last 3 laps for each driver from a .npz file.
    Filters files based on the year: skips 2024 in train mode, processes only 2024 in non-train mode.

    Args:
        npz_file_path (str): Path to the .npz file.
        train (bool): Flag to indicate if the function is in train mode.
    
    Returns:
        pd.DataFrame or None: Filtered DataFrame with the last 3 laps for each driver, 
                              or None if the file is skipped or has no data.
    """
    # Load the data from the .npz file
    data = np.load(npz_file_path, allow_pickle=True)["data"]

    # Handle train and non-train conditions
    if train:
        # Skip files from the 2024 season
        if npz_file_path.split('/')[-1].startswith('2024'):
            print(f"Skipping file from 2024 season: {npz_file_path}")
            return None
    else:
        # Process only files from the 2024 season
        if not npz_file_path.split('/')[-1].startswith('2024'):
            print(f"Skipping file not from 2024 season: {npz_file_path}")
            return None

    # If the dataset is empty, print a message and skip
    if data.shape[0] == 0:
        print(f"No data available in file: {npz_file_path.split('/')[-1]}. This driver did not participate in the race.")
        return None

    # Define the column names for the dataset
    columns = [
        'Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime',
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
        'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'Compound_x', 'TyreLife_x', 'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate', 'TrackStatus',
        'Position', 'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate', 'Compound_y',
        'TyreLife_y', 'TimeXY', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
        'WindDirection', 'WindSpeed', 'Date', 'SessionTime', 'DriverAhead', 'DistanceToDriverAhead',
        'Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source', 'Distance',
        'RelativeDistance', 'Status', 'X', 'Y', 'Z', 'Year', 'Event'
    ]

    # Create a DataFrame from the loaded data
    df = pd.DataFrame(data, columns=columns)

    # Find the last three laps using LapNumber
    unique_lap_numbers = sorted(df['LapNumber'].unique(), reverse=True)  # Get unique lap numbers in descending order
    last_3_laps_numbers = unique_lap_numbers[:3]  # Select the last 3 lap numbers

    # Filter the DataFrame for the last 3 laps
    last_3_laps = df[df['LapNumber'].isin(last_3_laps_numbers)]

    # Add the failure column
    failure = npz_file_path.split('_')[4].split(".")[0]
    last_3_laps['Failure'] = failure

    return last_3_laps


# main
if __name__ == "__main__":

    # Path to the folder and all file list
    folder_path = "../Dataset/OnlyFailuresByDriver/npz_failures"
    all_files = []

    # Collect all .npz files from the folder
    for f in os.listdir(folder_path):
        filepath = os.path.join(folder_path, f)
        all_files.append(filepath)

    all_files = sorted(all_files)

    final_failure = []

    # Set train or test mode
    train_mode = False  # Set to False for test mode

    # Process each file
    for file_path in all_files:
        print(f"Processing {file_path.split('/')[3]}...")
        last_3_laps = extract_last_3_laps(file_path, train=train_mode)
        if last_3_laps is not None:  # Only append valid results
            final_failure.append(last_3_laps)

    # Combine all DataFrames and sort by Driver and LapNumber
    final_failure_df = pd.concat(final_failure)

    # Normalize the data
    normClass = Normalizer(pit_stops=True)
    norm_final_failure = normClass.normalize_data(final_failure_df)

    # Save the dataset based on the mode (train or test)
    dataset_type = "train" if train_mode else "test"
    output_file = f'../Dataset/OnlyFailuresByDriver/npz_failures_MinMaxScaler_normalized_{dataset_type}.npz'
    np.savez_compressed(output_file, data=norm_final_failure)

    print(f"Dataset saved as: {output_file}")