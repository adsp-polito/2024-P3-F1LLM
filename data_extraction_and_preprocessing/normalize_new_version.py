import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

def convert_time_to_seconds(df, col):
    df[f'{col}_in_ms'] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds() * 1000
    df.drop(columns=[col], inplace=True)
    
def map_features(df):
    df = df.dropna(subset=['Team'])
    df = df.dropna(subset=['Event'])
    map_dict = {
        'Team': {
            'Ferrari': 1,
            'RB': 2,
            'Toro Rosso': 2,
            'AlphaTauri': 2,
            'Racing Point': 3,
            'Aston Martin': 3,
            'Alpine': 4,
            'Renault': 4,
            'Alfa Romeo Racing': 5,
            'Alfa Romeo': 5,
            'Kick Sauber': 5,
            'Williams': 6,
            'Haas F1 Team': 7,
            'Mercedes': 8,
            'Red Bull Racing': 9,
            'McLaren': 10,
        },

        'Event': {
            'Canadian Grand Prix': 1,
            'Belgian Grand Prix': 2,
            'British Grand Prix': 3,
            '70th Anniversary Grand Prix': 3,  # same id as british grand prix
            'São Paulo Grand Prix': 4,
            'Azerbaijan Grand Prix': 5,
            'Mexican Grand Prix': 6,
            'Russian Grand Prix': 7,
            'French Grand Prix': 8,
            'United States Grand Prix': 9,
            'Italian Grand Prix': 10,
            'Brazilian Grand Prix': 11,
            'Mexico City Grand Prix': 12,
            'Monaco Grand Prix': 13,
            'Qatar Grand Prix': 14,
            'Bahrain Grand Prix': 15,
            'Sakhir Grand Prix': 15,  # same id as 'Bahrain Grand Prix'
            'Portuguese Grand Prix': 16,
            'Singapore Grand Prix': 17,
            'Miami Grand Prix': 18,
            'Las Vegas Grand Prix': 19,
            'Abu Dhabi Grand Prix': 20,
            'Tuscan Grand Prix': 21,
            'German Grand Prix': 22,
            'Hungarian Grand Prix': 23,
            'Saudi Arabian Grand Prix': 24,
            'Australian Grand Prix': 25,
            'Chinese Grand Prix': 26,
            'Eifel Grand Prix': 27,
            'Austrian Grand Prix': 28,
            'Styrian Grand Prix': 28,  # same id as 'Austrian Grand Prix'
            'Japanese Grand Prix': 29,
            'Spanish Grand Prix': 30,
            'Turkish Grand Prix': 31,
            'Emilia Romagna Grand Prix': 32,
            'Dutch Grand Prix': 33,
        },

        'Compound': {
            'SOFT': 1,
            'MEDIUM': 2,
            'HARD': 3,
            'INTERMEDIATE': 4,
            'WET': 5
        },

        'DRS'   : {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
            9: False,
            10: True,
            11: True,
            12: True,
            13: True,
            14: True
        },

        'Failure':{
            'Others': 0,
            'BrakingSystem': 1,
            'Engine': 2,
            'PowerUnit': 3,
            'CoolingSystem': 4,
            'SuspensionandDrive': 5,
            'AerodynamicsandTyres': 6,
            'TransmissionandGearbox': 7,
        },
    }
    df = df.dropna(subset=['Compound'])
    df = df.dropna(subset=['TrackStatus'])
    df['Team'] = df['Team'].map(map_dict['Team'])
    df['Event'] = df['Event'].map(map_dict['Event'])
    df['Compound'] = df['Compound'].map(map_dict['Compound'])
    df['DRS'] = df['DRS'].map(map_dict['DRS'])
    if 'Failure' in df.columns:
        df['Failure'] = df['Failure'].map(map_dict['Failure'])
    df = df.dropna(subset=['DRS'])
    return df

def normalize_data(df, scaler_type="MinMaxScaler", filtered = True, driver_failures = []):
    print('Preprocessing data...')
    start_time = time.time()
    
    # Rename 'Compound_x' to 'Compound' and 'TyreLife_x' to 'TyreLife'
    if 'Compound_x' in df.columns and 'TyreLife_x' in df.columns:
        df.rename(columns={'Compound_x': 'Compound'}, inplace=True)
        df.rename(columns={'TyreLife_x': 'TyreLife'}, inplace=True)

    # List of time columns to convert
    time_columns = [
        'SessionTime',
        'Time',
        'PitOutTime',
        'PitInTime'
    ]

    # Convert all time columns to seconds
    for col in time_columns:
        if col in df.columns:
            convert_time_to_seconds(df, col)

    # Columns to normalize
    numerical_cols = [
        'SessionTime_in_ms',
        'Time_in_ms',
        # 'LapTime_in_ms',
        'LapNumber',
        'Position',
        'Speed',
        'AirTemp',
        'Humidity',
        'Pressure',
        'TrackTemp',
        'WindDirection',
        'WindSpeed',
        'DistanceToDriverAhead',
        'RPM',
        'nGear',
        'Throttle',
        'X', 
        'Y', 
        'Z', 
        'Distance', 
        'TyreLife',
    ]

    df.dropna(subset=['Time_in_ms'], inplace=True)
    # df.dropna(subset=['LapTime_in_ms'], inplace=True)

    if filtered:
        # Compute the set of pit stops for each driver
        driver_pit_laps = {
        driver: set(df[(df['DriverNumber'] == driver) & (df['PitInTime_in_ms'].notna())]['LapNumber'])
        .union(df[(df['DriverNumber'] == driver) & (df['PitOutTime_in_ms'].notna())]['LapNumber'])
        for driver in df['DriverNumber'].unique()}

        # Filter dataset using the precomputed mapping
        df = df[~df.apply(
            lambda row: row['LapNumber'] in driver_pit_laps[row['DriverNumber']],
            axis=1)]
        
        if 'IsAccurate' in df.columns:
            df = df[df['IsAccurate'] == True]
    else:
        #remove when driver is in the list of failures
        if len(driver_failures) != 0:
            df = df[~df['DriverNumber'].isin(driver_failures)]
        

    # One-hot encode 'Compound'
    if 'Compound' in df.columns:
        df = df[df['Compound'] != 'UNKNOWN'] # remove UNKNOWN compound records

    # Map Team and Event
    df = map_features(df)

    # Columns to drop (irrelevant or redundant)
    drop_cols = [
        'Time_x', 'Driver', 'DriverNumber', 'Stint', 'PitOutTime_in_ms', 'PitInTime_in_ms',
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
        'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'FreshTyre', 'LapStartTime', 'LapStartDate', 'Deleted', 'DeletedReason', 'FastF1Generated',
        'IsAccurate', 'Compound_y', 'TyreLife_y', 'Time_y', 'Date', 'SessionTime', 'Source',
        'RelativeDistance', 'Status', 'Year', 'LapTime', 'DriverAhead'
    ]

    # Drop irrelevant columns if they exist in the dataset
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Handle DistanceToDriverAhead

    if 'Position' in df.columns:
        df.loc[df['Position'] == 1, 'DistanceToDriverAhead'] = 0
        df = df.dropna(subset=['Position'])
        
    df['TrackStatus'] = df['TrackStatus'].replace('', np.nan)
    df = df.dropna(subset=['TrackStatus'])

    df['DistanceToDriverAhead'] = df['DistanceToDriverAhead'].astype(float).replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=['DistanceToDriverAhead'], inplace=True)

    # Preprocessing pipeline
    if scaler_type == 'StandardScaler':
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),  # Standardize numerical features
            ],
            remainder='passthrough'  # Retain non-normalized features as-is
        )
    elif scaler_type == 'MinMaxScaler':
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_cols),  # Normalize numerical features
            ],
            remainder='passthrough'  # Retain non-normalized features as-is
        )

    # Ensure all boolean columns are converted to int (otherwise it cause a problem)
    df = df.apply(lambda boolean_col: boolean_col.map({True: 1, False: 0}) if boolean_col.dtypes == 'bool' else boolean_col)

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)

    processed_data = processed_data.astype(np.float32) # Convert to float32 to reduce memory

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"Preprocessing took {elapsed_time:.2f} minutes.")
    return processed_data

def preprocessing_and_normalization(input_folder_path, output_folder_path = "normalized", scaler_type="MinMaxScaler", save_to_file=True):

    all_columns = [
        'Time_x', 'Driver', 'DriverNumber', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 'PitInTime',
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
        'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'Compound_x', 'TyreLife_x', 'FreshTyre', 'Team', 'LapStartTime', 'LapStartDate', 'TrackStatus',
        'Position', 'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate', 'Compound_y',
        'TyreLife_y', 'Time_y', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp',
        'WindDirection', 'WindSpeed', 'Date', 'SessionTime', 'DriverAhead', 'DistanceToDriverAhead',
        'Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS', 'Source', 'Distance',
        'RelativeDistance', 'Status', 'X', 'Y', 'Z', 'Year', 'Event'
    ]

    for file in os.listdir(input_folder_path):
        
        year = int(file.split('_')[0])
        event_name = file.split('_')[1].split('.')[0]

        # Load the data
        print(f'Loading {event_name} of {year}...')
        data_path = os.path.join(input_folder_path, file)
        np_data = np.load(data_path, allow_pickle=True)['data']
        print(f'Loaded! Converting to dataframe...')
        df = pd.DataFrame(np_data, columns=all_columns)
        print(f'Done!')
        
        failures_df = pd.read_csv('../Dataset/19-24_all_events_anomalies_new.csv')
        mask = (failures_df['EventName'].str.replace(' ', '') == event_name) & (failures_df['Year'] == year)
        filter_year_event = failures_df[mask]
        driver_failures = list(filter_year_event["DriverNumber"].reset_index(drop=True))
        
        normalized_data = normalize_data(df, scaler_type, driver_failures)
        np.savez_compressed(
                f'{output_folder}/normalized_data/{year}_{event_name}_{scaler_type}_normalized_complete.npz',
                data=normalized_data
            )
        
        # Normalize foreach driver
        print(driver_failures)
        for driver in driver_failures:
            df['DriverNumber'] = df['DriverNumber'].astype(int)
            driver_df = df[df['DriverNumber'] == int(driver)].copy()
            if not driver_df.empty:
                normalized_data_driver = normalize_data(driver_df, scaler_type, filtered=False)
            else:
                print(f'Driver {driver} did not start the race.')
            print(f'Saving cleaned data for {driver}...')
            np.savez_compressed(
                f'{output_folder}/normalized_data_by_driver/{year}_{event_name}_{driver}_{scaler_type}_normalized_complete_wPits.npz',
                data=normalized_data_driver
            )


if "__main__" == __name__:
    output_folder = 'D:/F1LLM_Datasets/npz_normalized_new'
    if not os.path.exists(f'{output_folder}'):
        os.makedirs(f'{output_folder}')
        
    if not os.path.exists(f'{output_folder}/normalized_data_by_driver'):
        os.makedirs(f'{output_folder}/normalized_data_by_driver')
        
    if not os.path.exists(f'{output_folder}/normalized_data'):
        os.makedirs(f'{output_folder}/normalized_data')

    input_folder = 'D:/F1LLM_Datasets/npz_all_telemetry_data'
    for year in [2022]:
        year_folder = os.path.join(input_folder, str(year))
        preprocessing_and_normalization(input_folder_path=year_folder)