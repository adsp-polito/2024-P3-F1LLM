import time
from array import array

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import os
import torch
from torch import nn


def convert_time_to_seconds(df, col):
    """
    Converts a timestamp string (e.g., '00:00.557') to seconds as float.
    Args:
        time_str (str): Time string in 'HH:MM:SS.sss' or 'MM:SS.sss' format.
    Returns:
        float: Time in seconds.
    """
    df[f'{col}_in_ms'] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds() * 1000
    df.drop(columns=[col], inplace=True)


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

def get_embedding_dim(num_categories):
    return min(50, (num_categories // 2) + 1)

def map_features(df):
    map_dict = {
        'Team': {
            'Ferrari': 1,
            'Racing Point': 2,
            'Alpine': 3,
            'Alfa Romeo Racing': 4,
            'Williams': 5,
            'RB': 6,
            'AlphaTauri': 7,
            'Haas F1 Team': 8,
            'Renault': 9,
            'Toro Rosso': 10,
            'Alfa Romeo': 11,
            'Mercedes': 12,
            'Aston Martin': 13,
            'Red Bull Racing': 14,
            'McLaren': 15,
            'Kick Sauber': 0
        },
        'Event': {
            'Canadian Grand Prix': 1,
            'Belgian Grand Prix': 2,
            'British Grand Prix': 3,
            'São Paulo Grand Prix': 4,
            '70th Anniversary Grand Prix': 3,  # same id as british grand prix
            'Pre-Season Testing': 6,  # to be removed
            'Azerbaijan Grand Prix': 7,
            'Mexican Grand Prix': 8,
            'Russian Grand Prix': 9,
            'French Grand Prix': 10,
            'United States Grand Prix': 11,
            'Italian Grand Prix': 12,
            'Brazilian Grand Prix': 13,
            'Mexico City Grand Prix': 14,
            'Pre-Season Test 1': 15,  # to be removed
            'Monaco Grand Prix': 16,
            'Qatar Grand Prix': 17,
            'Bahrain Grand Prix': 18,
            'Portuguese Grand Prix': 19,
            'Singapore Grand Prix': 20,
            'Miami Grand Prix': 21,
            'Las Vegas Grand Prix': 22,
            'Abu Dhabi Grand Prix': 23,
            'Pre-Season Track Session': 24,  # to be removed
            'Pre-Season Test 2': 25,  # to be removed
            'Sakhir Grand Prix': 18,  # same id as 'Bahrain Grand Prix'
            'Tuscan Grand Prix': 27,
            'German Grand Prix': 28,
            'Hungarian Grand Prix': 29,
            'Saudi Arabian Grand Prix': 30,
            'Australian Grand Prix': 31,
            'Chinese Grand Prix': 32,
            'Eifel Grand Prix': 33,
            'Austrian Grand Prix': 34,
            'Japanese Grand Prix': 35,
            'Spanish Grand Prix': 36,
            'Turkish Grand Prix': 37,
            'Pre-Season Test': 38,  # to be removed
            'Emilia Romagna Grand Prix': 39,
            'Dutch Grand Prix': 0,

            'Styrian Grand Prix': 34,  # same id as 'Austrian Grand Prix'
        },
        'Compound': {
            'SOFT': 1,
            'MEDIUM': 2,
            'HARD': 3,
            'INTERMEDIATE': 4,
            'WET': 0
        },
        'TrackStatus': {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
        },
        'DRS'   : {
            0: 0,
            1: 0,
            2: np.nan,
            3: np.nan,
            4: np.nan,
            5: np.nan,
            6: np.nan,
            7: np.nan,
            8: np.nan,
            9: np.nan,
            10: 3,
            11: 3,
            12: 3,
            13: 3,
            14: 3
        }
    }

    df = df.dropna(subset=['Team'])
    df = df.dropna(subset=['Event'])
    df = df.dropna(subset=['Compound'])
    df = df.dropna(subset=['TrackStatus'])
    df = df.dropna(subset=['DRS'])

    df['Team'] = df['Team'].map(map_dict['Team'])
    df['Event'] = df['Event'].map(map_dict['Event'])
    df['Compound'] = df['Compound'].map(map_dict['Compound'])
    df['TrackStatus'] = df['TrackStatus'].map(map_dict['TrackStatus'])
    df['DRS'] = df['DRS'].map(map_dict['DRS'])
    df = df.dropna(subset=['DRS'])

    # Define embedding dimensions for each categorical column
    num_compounds = get_embedding_dim(len(map_dict['Compound']))
    num_teams = get_embedding_dim(len(map_dict['Team']))
    num_events = get_embedding_dim(len(map_dict['Event']) - 8)
    num_track_status = get_embedding_dim(len(map_dict['TrackStatus']))
    num_DRS = get_embedding_dim(len(map_dict['DRS']))

    compound_embedding_dim = 3
    team_embedding_dim = 6
    event_embedding_dim = 8
    track_status_embedding_dim = 3
    drs_embedding_dim = 3

    # Create embedding layers
    compound_embedding = nn.Embedding(num_compounds, compound_embedding_dim)
    team_embedding = nn.Embedding(num_teams, team_embedding_dim)
    event_embedding = nn.Embedding(num_events, event_embedding_dim)
    track_status_embedding = nn.Embedding(num_track_status, track_status_embedding_dim)
    drs_embedding = nn.Embedding(num_DRS, drs_embedding_dim)

    return df, compound_embedding, team_embedding, event_embedding, track_status_embedding, drs_embedding


def preprocess_data(df):
    """
    Preprocesses the data without normalization.

    Args:
        df (DataFrame): Input dataset containing raw telemetry and race data.

    Returns:
        np.array: Preprocessed dataset as a NumPy array, with irrelevant features removed.
    """
    print('Preprocessing data...')
    start_time = time.time()

    # Rename 'Compound_x' to 'Compound' and 'TyreLife_x' to 'TyreLife'
    if 'Compound_x' in df.columns and 'TyreLife_x' in df.columns:
        df.rename(columns={'Compound_x': 'Compound', 'TyreLife_x': 'TyreLife'}, inplace=True)

    # List of time columns to convert
    time_columns = [
        'Time', 'LapTime'  # , 'PitOutTime', 'PitInTime'
    ]

    # Convert all time columns to seconds
    for col in time_columns:
        if col in df.columns:
            convert_time_to_seconds(df, col)

    # Ensure 'DriverAhead' is numeric or NaN
    if 'DriverAhead' in df.columns:
        df['DriverAhead'] = df['DriverAhead'].apply(lambda x: x if pd.isna(x) else int(x))

    # Handle pit stop time as a derived feature
    # if 'PitOutTime_in_ms' in df.columns and 'PitInTime_in_ms' in df.columns:
    #     df['PitInTime_in_ms'] = df['PitInTime_in_ms'].fillna(0).astype(float)
    #     df['PitOutTime_in_ms'] = df['PitOutTime_in_ms'].fillna(0).astype(float)
    #     df['PitStopTime_in_ms'] = (df['PitOutTime_in_ms'] - df['PitInTime_in_ms']).astype(float)
    #     df.drop(columns=['PitOutTime_in_ms', 'PitInTime_in_ms'], errors='ignore', inplace=True)

    # Remove rows where 'LapNumber' corresponds to laps with a pit stop
    filtered_dfs = []
    for driver, group in df.groupby('DriverNumber'):
        laps_with_pitin = group[group['PitInTime'].notna()]['LapNumber'].unique()
        laps_with_pitout = group[group['PitOutTime'].notna()]['LapNumber'].unique()

        filtered_group = group[
            ~group['LapNumber'].isin(laps_with_pitin) &
            ~group['LapNumber'].isin(laps_with_pitout)
            ]
        filtered_dfs.append(filtered_group)

    # Combine back into one DataFrame
    df = pd.concat(filtered_dfs, ignore_index=True)

    if 'Compound' in df.columns:
        df = df[df['Compound'] != 'UNKNOWN']  # Remove UNKNOWN compound records

    # Define a list of pre-season events
    pre_season_events = [
        "Pre-Season Testing", "Pre-Season Test 1", "Pre-Season Test 2",
        "Pre-Season Track Session", "Pre-Season Test"
    ]

    # Drop rows with Event in pre-season events
    df = df[~df['Event'].isin(pre_season_events)]

    # Map Team and Event
    df, compound_embedding, team_embedding, event_embedding, track_status_embedding, drs_embedding = map_features(df)

    # Columns to drop (irrelevant or redundant)
    drop_cols = [
        'Driver', 'DriverNumber', 'Stint', 'PitOutTime', 'PitInTime',
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
        'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'FreshTyre', 'LapStartTime', 'LapStartDate', 'Deleted', 'DeletedReason', 'FastF1Generated',
        'IsAccurate', 'Compound_y', 'TyreLife_y', 'TimeXY', 'Date', 'SessionTime', 'Source',
        'RelativeDistance', 'Status', 'Year'
    ]

    # Drop irrelevant columns if they exist in the dataset
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore', inplace=True)

    # Handle DistanceToDriverAhead
    if 'Position' in df.columns:
        df.loc[df['Position'] == 1, 'DistanceToDriverAhead'] = 0
        df.loc[df['Position'] == 1, 'DriverAhead'] = 0

    if 'LapTime_in_ms' in df.columns:
        df.dropna(subset=['LapTime_in_ms'], inplace=True)

    if 'DriverAhead' in df.columns:
        df.dropna(subset=['DriverAhead'], inplace=True)
        df['DriverAhead'] = df['DriverAhead'].astype('int8')

    if 'DistanceToDriverAhead' in df.columns:
        df['DistanceToDriverAhead'] = df['DistanceToDriverAhead'].replace([np.inf, -np.inf], np.nan)
        df.dropna(subset=['DistanceToDriverAhead'], inplace=True)

    # Ensure all boolean columns are converted to int
    df = df.apply(lambda col: col.map({True: 1, False: 0}) if col.dtypes == 'bool' else col)

    compound_array = df['Compound'].to_numpy(dtype=np.int8)
    df.drop(columns=['Compound'], errors='ignore', inplace=True)
    team_array = df['Team'].to_numpy(dtype=np.int8)
    df.drop(columns=['Team'], errors='ignore', inplace=True)
    event_array = df['Event'].to_numpy(dtype=np.int8)
    df.drop(columns=['Event'], errors='ignore', inplace=True)
    track_status_array = df['TrackStatus'].to_numpy(dtype=np.int8)
    df.drop(columns=['TrackStatus'], errors='ignore', inplace=True)
    drs_array = df['DRS'].to_numpy(dtype=np.int8)
    df.drop(columns=['DRS'], errors='ignore', inplace=True)

    numeric_cols = ['LapNumber', 'TyreLife', 'Position', 'AirTemp', 'Humidity', 'Pressure', 'TrackTemp',
                    'WindDirection', 'WindSpeed', 'RPM', 'Speed', 'nGear', 'Throttle', 'Distance', 'X', 'Y', 'Z']
    
    print(f"Shape of DataFrame before scaling: {df.shape}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Shape of df[numeric_cols]: {df[numeric_cols].shape}")


    scaler = StandardScaler()

    # Fit and transform the numerical data
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save the scaler for later use (e.g., when normalizing test data)
    joblib.dump(scaler, 'StandardScaler.pkl')

    # Convert DataFrame to NumPy array
    processed_data = df.to_numpy(dtype=np.float32)

    # Extract categorical columns
    compound_indices = torch.tensor(compound_array, dtype=torch.long)  # Compound_x
    team_indices = torch.tensor(team_array, dtype=torch.long)  # Team
    event_indices = torch.tensor(event_array, dtype=torch.long)  # Event
    track_status_indices = torch.tensor(track_status_array, dtype=torch.long)
    drs_indices = torch.tensor(drs_array, dtype=torch.long)

    # Embed the categorical columns
    compound_embedded = compound_embedding(compound_indices)
    team_embedded = team_embedding(team_indices)
    event_embedded = event_embedding(event_indices)
    track_status_embedded = track_status_embedding(track_status_indices)
    drs_embedded = drs_embedding(drs_indices)

    other_columns = torch.tensor(processed_data, dtype=torch.float)  # Remaining columns
    embedded_data = torch.cat([other_columns, compound_embedded, team_embedded, event_embedded, track_status_embedded, drs_embedded], dim=1)
    embedded_array = embedded_data.detach().numpy()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"Preprocessing took {elapsed_time:.2f} minutes.")

    return embedded_array


all_columns = [
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

print(f'Loaded! Converting to dataframe...')

dtype_dict = {
    "Driver": 'category',
    "DriverNumber": 'category',
    "LapNumber": 'int8',
    "Compound_x": 'category',
    "Compound_y": 'category',
    "TyreLife_x": 'int8',
    "TyreLife_y": 'int8',
    "Team": 'category',
    "TrackStatus": 'category',
    "Position": 'int8',
    "AirTemp": 'float16',
    "Humidity": 'float16',
    "Pressure": 'float16',
    "Rainfall": bool,
    "TrackTemp": 'float16',
    "WindDirection": 'float16',
    "WindSpeed": 'float16',
    "DistanceToDriverAhead": float,
    "RPM": 'int16',
    "Speed": 'int16',
    "nGear": 'int8',
    "Throttle": 'int8',
    "Brake": bool,
    "DRS": 'category',
    "Distance": 'float32',
    "X": 'float16',
    "Y": 'float16',
    "Z": 'float16',
    "Event": 'category',

    'PitOutTime': object,
    'PitInTime': object,
    "LapTime": object,
    "Time": object,

    'IsPersonalBest': bool,
    'FreshTyre': bool,
    'FastF1Generated': bool,
    'IsAccurate': bool,
    'Source': 'category',
    'Deleted': bool,
    'DeletedReason': 'category',
    'Year': 'int16',
    'RelativeDistance': float
}

for year in range(2021, 2025):
    for file in os.listdir(f'D:/NumpyData/{year}/'):

        print(f'Loading {file}...')
        file_path = os.path.join(f'D:/NumpyData/{year}/', file)
        np_data = np.load(file_path, allow_pickle=True)['data']
        df = pd.DataFrame(np_data, columns=all_columns)
        df = df.astype(dtype_dict)

        print(f'Done!')

        cleaned_data = preprocess_data(df)

        print('Saving cleaned data...')
        stime = time.time()
        eventName = file.split('_')[1]
        np.savez_compressed(f'anomaly_embedded/{year}_{eventName}_embedded.npz', data=cleaned_data)
        etime = time.time()
        print(f'Done in {(etime - stime) / 60:.2f} minutes')
