import numpy as np
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from data_extraction_and_preprocessing.normalize_npz import Normalizer
import os


def extract_last_3_laps(df):

    # Find the last three laps using LapNumber
    unique_lap_numbers = sorted(df['LapNumber'].unique(), reverse=True)  # Get unique lap numbers in descending order
    last_3_laps_numbers = unique_lap_numbers[:3]  # Select the last 3 lap numbers

    # Filter the DataFrame for the last 3 laps
    last_3_laps = df[df['LapNumber'].isin(last_3_laps_numbers)]

    return last_3_laps


# main
if __name__ == "__main__":

    # Normalize the data
    normClass = Normalizer(pit_stops=True)

    folder_path = '../Dataset/OnlyFailuresByDriver/npz_failures'
    train_mode = True

    driver = None # set a value or None
    event = None # set a value or None

    df_columns = [
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
                print(f'Driver: {driver_file}, Event: {event_file}')
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

            norm_df = normClass.normalize_data(df)
            all_data.append(norm_df)
            print(f'Shape: {norm_df.shape}')
            print(f'Loaded {f}!, All data: {len(all_data)}')
        else:
            print(f"File {f} did not contain valid data.")
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