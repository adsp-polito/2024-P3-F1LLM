import time
import numpy as np
import pandas as pd
from numba.cuda.libdevice import half2float
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os


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
            'Kick Sauber': 16
        }, 
        'Event': {
            'Canadian Grand Prix': 1,
            'Belgian Grand Prix': 2,
            'British Grand Prix': 3,
            'São Paulo Grand Prix': 4,
            '70th Anniversary Grand Prix': 3, # same id as british grand prix
            'Pre-Season Testing': 6,    # to be removed
            'Azerbaijan Grand Prix': 7,
            'Mexican Grand Prix': 8,
            'Russian Grand Prix': 9,
            'French Grand Prix': 10,
            'United States Grand Prix': 11,
            'Italian Grand Prix': 12,
            'Brazilian Grand Prix': 13,
            'Mexico City Grand Prix': 14,
            'Pre-Season Test 1': 15,    # to be removed
            'Monaco Grand Prix': 16,
            'Qatar Grand Prix': 17,
            'Bahrain Grand Prix': 18,
            'Portuguese Grand Prix': 19,
            'Singapore Grand Prix': 20,
            'Miami Grand Prix': 21,
            'Las Vegas Grand Prix': 22,
            'Abu Dhabi Grand Prix': 23,
            'Pre-Season Track Session': 24,  # to be removed
            'Pre-Season Test 2': 25,    # to be removed
            'Sakhir Grand Prix': 18,    # same id as 'Bahrain Grand Prix'
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
            'Dutch Grand Prix': 40,

            'Styrian Grand Prix': 34, # same id as 'Austrian Grand Prix'
        },
        'Compound': {
            'SOFT': 1,
            'MEDIUM': 2,
            'HARD': 3,
            'INTERMEDIATE': 4,
            'WET': 5
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
    
    df['Team'] = df['Team'].map(map_dict['Team'])
    df['Event'] = df['Event'].map(map_dict['Event'])
    df['Compound'] = df['Compound'].map(map_dict['Compound'])
    df['DRS'] = df['DRS'].map(map_dict['DRS'])
    
    df = df.dropna(subset=['DRS'])

    return df


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

    print('Preprocessing data...')
    start_time = time.time()

    # Rename 'Compound_x' to 'Compound' and 'TyreLife_x' to 'TyreLife'
    if 'Compound_x' in df.columns and 'TyreLife_x' in df.columns:
        df.rename(columns={'Compound_x': 'Compound'}, inplace=True)
        df.rename(columns={'TyreLife_x': 'TyreLife'}, inplace=True)

    # List of time columns to convert
    time_columns = [
        'Time', 'LapTime', 'PitOutTime', 'PitInTime'
    ]

    # Convert all time columns to seconds
    for col in time_columns:
        if col in df.columns:
            # df[col] = df[col].apply(convert_time_to_seconds)
            convert_time_to_seconds(df, col)


    # Columns to normalize
    numerical_cols = [
        'Time_in_ms',
        'LapTime_in_ms',
        'LapNumber',
        'Position',
        'Speed',
        'AirTemp',
        'Humidity',
        'Pressure',
        # 'Rainfall',   # boolean
        'TrackTemp',
        'WindDirection',
        'WindSpeed',
        'DistanceToDriverAhead',
        'RPM',
        'nGear',
        'Throttle', 
        # 'DRS',        # categorical
        'X', 
        'Y', 
        'Z', 
        'Distance', 
        'TyreLife', 
        #'TrackStatus'  # categorical
    ]

    df.dropna(subset=['Time_in_ms'], inplace=True)
    df.dropna(subset=['LapTime_in_ms'], inplace=True)

    # Handle pit stop time as a derived feature
    # if 'PitOutTime_in_ms' in df.columns and 'PitInTime_in_ms' in df.columns:
    #     df['PitInTime_in_ms'] = df['PitInTime_in_ms'].fillna(0).astype(float)
    #     df['PitOutTime_in_ms'] = df['PitOutTime_in_ms'].fillna(0).astype(float)
    #     df['PitStopTime_in_ms'] = (df['PitOutTime_in_ms'] - df['PitInTime_in_ms']).astype(float)
    #     df = df.drop(columns=['PitOutTime_in_ms', 'PitInTime_in_ms'], errors='ignore')

    # drivers = df['DriverNumber'].unique()

    # print(f'Number of drivers: {len(drivers)}')
    # print(f'Number of laps: {df["LapNumber"].unique()}')

    # for driver in drivers:
    #     lapsPitIn = df[df['PitInTime_in_ms'].notna()]['LapNumber'].unique()
    #     lapsPitOut = df[df['PitOutTime_in_ms'].notna()]['LapNumber'].unique()

    #     df = df[~df['DriverNumber'].isin([driver]) & (df['PitInTime_in_ms'].isna(lapsPitIn))]
    #     df = df[~df['DriverNumber'].isin([driver]) & (df['PitOutTime_in_ms'].isna(lapsPitOut))]

    #     print(f'Number of laps for driver {driver} after removing pit stops: {df["LapNumber"].unique()}')

    # print(f'Number of laps after removing pit stops: {df["LapNumber"].unique()}')

    print(f'Total number of laps before removing all pit stops: {df["LapNumber"].nunique()}')
    print(f'Dataframe shape before removing all pit stops: {df.shape}')

    # Compute the set of pit stops for each driver
    driver_pit_laps = {
    driver: set(df[(df['DriverNumber'] == driver) & (df['PitInTime_in_ms'].notna())]['LapNumber'])
    .union(df[(df['DriverNumber'] == driver) & (df['PitOutTime_in_ms'].notna())]['LapNumber'])
    for driver in df['DriverNumber'].unique()
    }

    # Filter dataset using the precomputed mapping
    df = df[~df.apply(
    lambda row: row['LapNumber'] in driver_pit_laps[row['DriverNumber']],
    axis=1
    )]

    print(f'Total number of laps after removing all pit stops: {df["LapNumber"].nunique()}')
    print(f'Dataframe shape after removing all pit stops: {df.shape}')

    # One-hot encode 'Compound'
    if 'Compound' in df.columns:
        df = df[df['Compound'] != 'UNKNOWN'] # remove UNKNOWN compound records
        # one_hot = pd.get_dummies(df['Compound'], prefix='Compound')
        # df = pd.concat([df, one_hot], axis=1).drop(columns=['Compound'])


    # Define a list of pre-season events
    pre_season_events = [
        "Pre-Season Testing", "Pre-Season Test 1", "Pre-Season Test 2", 
        "Pre-Season Track Session", "Pre-Season Test"
    ]

    # Drop rows with Event in pre-season events
    df = df[~df['Event'].isin(pre_season_events)]

    # Map Team and Event
    df = map_features(df)

    # Columns to drop (irrelevant or redundant)
    drop_cols = [
        'Driver', 'DriverNumber', 'Stint', 'PitOutTime_in_ms', 'PitInTime_in_ms',
        'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
        'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
        'FreshTyre', 'LapStartTime', 'LapStartDate', 'Deleted', 'DeletedReason', 'FastF1Generated',
        'IsAccurate', 'Compound_y', 'TyreLife_y', 'TimeXY', 'Date', 'SessionTime', 'Source',
        'RelativeDistance', 'Status', 'Year'
    ]

    print(f'columns before drop: {df.columns}\n')

    # Drop irrelevant columns if they exist in the dataset
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    print(f'columns after drop: {df.columns}\n')

    # Check existing columns in the dataset
    # numerical_cols = [col for col in numerical_cols if col in df.columns]

    # Handle DistanceToDriverAhead and DriverAhead
    df['DriverAhead'] = df['DriverAhead'].apply(lambda x: x if pd.isna(x) else int(x))

    if 'Position' in df.columns:
        df.loc[df['Position'] == 1, 'DistanceToDriverAhead'] = 0
        df.loc[df['Position'] == 1, 'DriverAhead'] = 0

    df = df.dropna(subset=['DriverAhead'])
    df['DriverAhead'] = df['DriverAhead'].astype('int8')
    df['DistanceToDriverAhead'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['DistanceToDriverAhead'], inplace=True)

    # Remove records where LapNumber is equal to 1 or 2
    # df = df[~df['LapNumber'].isin([1, 2])]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # Standardize numerical features
        ],
        remainder='passthrough'  # Retain non-normalized features as-is
    )

    # print("\n", df.dtypes, "\n")

    # Ensure all boolean columns are converted to int (otherwise it cause a problem)
    df = df.apply(lambda col: col.map({True: 1, False: 0}) if col.dtypes == 'bool' else col)

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)

    processed_data = processed_data.astype(np.float32) # Convert to float32 to reduce memory

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # Convert to minutes
    print(f"Preprocessing took {elapsed_time:.2f} minutes.")
    return processed_data


all_columns=[
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

folder_path = f'D:/NumpyData'

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
            "DistanceToDriverAhead": 'float32',
            "RPM": 'int16',
            "Speed": 'int16',
            "nGear": 'int8',
            "Throttle": 'int8',
            "Brake": bool,
            "DRS": 'int8',
            "Distance": 'float32',
            "X": 'float32',
            "Y": 'float32',
            "Z": 'float32',
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

for year in range(2019, 2020):
    file_path = os.path.join(folder_path, f'{year}')
    for file in os.listdir(file_path):
        print(f'Loading {file}...')

        data_path = os.path.join(file_path, file)
        np_data = np.load(data_path, allow_pickle=True)['data']
        event_name = file.split('_')[1]
        print(f'Loaded! Converting to dataframe...')

        df = pd.DataFrame(np_data, columns=all_columns)
        df = df.astype(dtype_dict)

        

        print(f'Done!')

        cleaned_data = normalize_data(df)

        print('Saving cleaned data...')
        stime = time.time()
        np.savez_compressed(f'AD_supernormalized/{year}_{event_name}__AD_supernormalized.npz', data=cleaned_data)
        etime = time.time()
        print(f'Done in {(etime - stime)/60:.2f} minutes')
