import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os


class Normalizer:
    def __init__(self, input_folder, output_folder, pit_stops=False, given_driver=None, given_year=None, given_event=None):
        self.pit_stops = pit_stops

        self.input_folder_path = input_folder
        self.output_folder_path = output_folder

        self.given_driver = given_driver
        self.given_year = given_year
        self.given_event = given_event

    def convert_time_to_seconds(self, df, col):
        """
        Converts a timestamp string (e.g., '00:00.557') to seconds as float.
        Args:
            time_str (str): Time string in 'HH:MM:SS.sss' or 'MM:SS.sss' format.
        Returns:
            float: Time in seconds.
        """
        df[f'{col}_in_ms'] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds() * 1000
        df.drop(columns=[col], inplace=True)

    def map_features(self, df):
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

                # 'Pre-Season Test': 0,  # to be removed
                # 'Pre-Season Track Session': 0,  # to be removed
                # 'Pre-Season Test 2': 0,    # to be removed
                # 'Pre-Season Testing': 0,  # to be removed
                # 'Pre-Season Test 1': 0,  # to be removed
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
                2: np.nan,
                3: np.nan,
                4: np.nan,
                5: np.nan,
                6: np.nan,
                7: np.nan,
                8: np.nan,
                9: np.nan,
                10: True,
                11: True,
                12: True,
                13: True,
                14: True
            },

            'Failure':{
                'Others': 0,
                'Braking System': 1,
                'Engine': 2,
                'Power Unit': 3,
                'Cooling System': 4,
                'Suspension and Drive': 5,
                'Aerodynamics and Tyres': 6,
                'Transmission and Gearbox': 7,
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
        if 'Failure' in df.columns:
            df['Failure'] = df['Failure'].map(map_dict['Failure'])
        
        df = df.dropna(subset=['DRS'])
        print('map superato!')
        return df

    def normalize_data(self, df, scaler_type="MinMaxScaler"):
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
                    :param scaler_type:
        """

        print('Preprocessing data...')
        start_time = time.time()

        # Rename 'Compound_x' to 'Compound' and 'TyreLife_x' to 'TyreLife'
        if 'Compound_x' in df.columns and 'TyreLife_x' in df.columns:
            df.rename(columns={'Compound_x': 'Compound'}, inplace=True)
            df.rename(columns={'TyreLife_x': 'TyreLife'}, inplace=True)

        # List of time columns to convert
        time_columns = [
            'SessionTime', 'Time', 'LapTime', 'PitOutTime', 'PitInTime'
        ]

        # Convert all time columns to seconds
        for col in time_columns:
            if col in df.columns:
                self.convert_time_to_seconds(df, col)

        # Columns to normalize
        numerical_cols = [
            'SessionTime_in_ms',
            'Time_in_ms',
            'LapTime_in_ms',
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
        df.dropna(subset=['LapTime_in_ms'], inplace=True)


        # Compute the set of pit stops for each driver
        if not self.pit_stops:
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

        # One-hot encode 'Compound'
        if 'Compound' in df.columns:
            df = df[df['Compound'] != 'UNKNOWN'] # remove UNKNOWN compound records
        print('inizio il map')
        # Map Team and Event
        df = self.map_features(df)

        # Columns to drop (irrelevant or redundant)
        drop_cols = [
            'Time_x', 'Driver', 'DriverNumber', 'Stint', 'PitOutTime_in_ms', 'PitInTime_in_ms',
            'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
            'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
            'FreshTyre', 'LapStartTime', 'LapStartDate', 'Deleted', 'DeletedReason', 'FastF1Generated',
            'IsAccurate', 'Compound_y', 'TyreLife_y', 'TimeXY', 'Date', 'SessionTime', 'Source',
            'RelativeDistance', 'Status', 'Year'
        ]

        # Drop irrelevant columns if they exist in the dataset
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Handle DistanceToDriverAhead and DriverAhead
        ('driverAhead1')
        df['DriverAhead'] = df['DriverAhead'].apply(lambda x: x if pd.isna(x) else int(x))

        if 'Position' in df.columns:
            df.loc[df['Position'] == 1, 'DistanceToDriverAhead'] = 0
            df.loc[df['Position'] == 1, 'DriverAhead'] = 0

        df = df.dropna(subset=['DriverAhead'])
        print('driverAhead2')
        df['DriverAhead'] = df['DriverAhead'].astype('int8')
        df['DistanceToDriverAhead'].replace([np.inf, -np.inf], np.nan, inplace=True)
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
        print('conversione boolean')
        df = df.apply(lambda boolean_col: boolean_col.map({True: 1, False: 0}) if boolean_col.dtypes == 'bool' else boolean_col)

        # Apply transformations
        processed_data = preprocessor.fit_transform(df)

        processed_data = processed_data.astype(np.float32) # Convert to float32 to reduce memory

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60  # Convert to minutes
        print(f"Preprocessing took {elapsed_time:.2f} minutes.")
        return processed_data

    def preprocessing_and_normalization(self, scaler_type="MinMaxScaler", normalize_by_driver=False):

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

        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        for year_folder in os.listdir(self.input_folder_path):
            year_folder_path = os.path.join(self.input_folder_path, year_folder)

            for file in os.listdir(year_folder_path):
                try:
                    if not file.split('_')[1].startswith('Pre'):

                        print(f'Loading {file}...')

                        data_path = os.path.join(year_folder_path, file)
                        np_data = np.load(data_path, allow_pickle=True)['data']

                        year = file.split('_')[0]
                        event_name = file.split('_')[1].split('.')[0]
                        print(f'Loaded! Converting to dataframe...')

                        df = pd.DataFrame(np_data, columns=all_columns)
                        df = df.astype(dtype_dict)

                        print(f'Done!')

                        if self.given_event is not None and self.given_year is not None and self.given_driver is not None:
                            if event_name == self.given_event or year == self.given_year:

                                for driver in df['Driver'].unique():

                                    # Filter the DataFrame for the current driver
                                    if driver == self.given_driver:
                                        driver_df = df[df['Driver'] == driver].copy()

                                        # Normalize the data for the current driver
                                        cleaned_data = self.normalize_data(driver_df, scaler_type)

                                        print(f'Saving cleaned data for {driver}...')

                                        # Save the cleaned and normalized data for the current driver
                                        np.savez_compressed(
                                            f'{self.output_folder_path}/test_data/selected_driver/{year}_{event_name}_{scaler_type}_{driver}_normalized_complete_wPits.npz',
                                            data=cleaned_data
                                        )

                        elif normalize_by_driver:
                            for driver in df['Driver'].unique():

                                # Filter the DataFrame for the current driver
                                driver_df = df[df['Driver'] == driver].copy()

                                # Normalize the data for the current driver
                                cleaned_data = self.normalize_data(driver_df, scaler_type)

                                print(f'Saving cleaned data for {driver}...')

                                # Save the cleaned and normalized data for the current driver
                                np.savez_compressed(
                                    f'{self.output_folder_path}/test_data/normalized_data_by_driver/{year}_{event_name}_{scaler_type}_{driver}_normalized_complete_wPits.npz',
                                    data=cleaned_data
                                )
                        else:
                            # Normalize the data for the entire event and year
                            cleaned_data = self.normalize_data(df, scaler_type)

                            print('Saving cleaned data...')
                            stime = time.time()

                            if not os.path.exists(f'{self.output_folder_path}/train_data'):
                                os.makedirs(f'{self.output_folder_path}/train_data')
                            if not os.path.exists(f'{self.output_folder_path}/'):
                                os.makedirs(f'{self.output_folder_path}/')

                            np.savez_compressed(
                                f'{self.output_folder_path}/{year}_{event_name}_{scaler_type}_normalized.npz',
                                data=cleaned_data)
                            etime = time.time()
                            print(f'Done in {(etime - stime) / 60:.2f} minutes')

                    else:
                        print('Skipping pre-season event...')

                except Exception as e:
                    print(f'Error processing {file}: {e}')

if __name__ == '__main__':

    input_folder = '../temp'   # EDIT THIS PATH
    output_folder = '../temp'  # EDIT THIS PATH

    if not os.path.exists(input_folder):
        print('Input folder does not exist. Exiting...')
        exit(1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    user_input = input('Select the operation to perform:\n '
                       '1. [TRAIN] Normalize ALL data by event and year REMOVING pit-stops, and driver with failures (saved separately for classification training)\n '
                       '2. [TRAIN] Normalize ALL data by event and year MAINTAINING pit-stops and REMOVING driver with failures (saved separately for classification training)\n '
                       '3. [TEST] Normalize data by driver, event and year MAINTAINING pit-stops\n '
                       '4. [TEST] Normalize a single file (driver, event, year)\n '
                       '--> ')

    if user_input == '1':

        if not os.path.exists(f'{output_folder}/train_data'):
            os.makedirs(f'{output_folder}/train_data')
        if not os.path.exists(f'{output_folder}/train_data/train_data_without_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_without_failures')
        if not os.path.exists(f'{output_folder}/train_data/train_data_with_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_with_failures')

        normalizer = Normalizer(input_folder, output_folder, pit_stops=False)
        normalizer.preprocessing_and_normalization()

    elif user_input == '2':

        if not os.path.exists(f'{output_folder}/train_data'):
            os.makedirs(f'{output_folder}/train_data')
        if not os.path.exists(f'{output_folder}/train_data/train_data_without_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_without_failures')
        if not os.path.exists(f'{output_folder}/train_data/train_data_with_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_with_failures')

        normalizer = Normalizer(input_folder, output_folder, pit_stops=True)
        normalizer.preprocessing_and_normalization()

    elif user_input == '3':

        if not os.path.exists(f'{output_folder}/test_data'):
            os.makedirs(f'{output_folder}/test_data')
        if not os.path.exists(f'{output_folder}/test_data/normalized_data_by_driver'):
            os.makedirs(f'{output_folder}/test_data/normalized_data_by_driver')

        normalizer = Normalizer(input_folder, output_folder, pit_stops=True)
        normalizer.preprocessing_and_normalization(normalize_by_driver=True)

    elif user_input == '4':

        driver = input('Enter the driver name (first 3 letters): ')
        driver = driver.upper()
        event = input('Enter the event name: ')
        year = input('Enter the year: ')

        if not os.path.exists(f'{output_folder}/test_data'):
            os.makedirs(f'{output_folder}/test_data')
        if not os.path.exists(f'{output_folder}/test_data/selected_driver'):
            os.makedirs(f'{output_folder}/test_data/selected_driver')

        normalizer = Normalizer(input_folder, output_folder, given_driver=driver, given_event=event, given_year=year, pit_stops=True)
        normalizer.preprocessing_and_normalization()
    else:
        print('Invalid input. Exiting...')
        exit(1)

