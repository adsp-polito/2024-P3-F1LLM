import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os


class Normalizer:
    def __init__(self, input_folder_path, output_folder_path, given_driver=None, given_year=None, given_event=None):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

        self.given_driver = given_driver
        self.given_year = self.given_year = int(given_year) if given_year is not None else None
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
            }
        }
        df = df.dropna(subset=['Compound'])
        df = df.dropna(subset=['TrackStatus'])
        
        df['Team'] = df['Team'].map(map_dict['Team'])
        df['Event'] = df['Event'].map(map_dict['Event'])
        df['Compound'] = df['Compound'].map(map_dict['Compound'])
        df['DRS'] = df['DRS'].map(map_dict['DRS'])
        if 'Failures' in df.columns:
            df['Failures'] = df['Failures'].map(map_dict['Failures'])
        
        df = df.dropna(subset=['DRS'])
        return df

    def normalize_data(self, df, scaler_type="MinMaxScaler", include_not_accurate_and_pit=False):
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
                    :param include_not_accurate_and_pit:
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
            'SessionTime',
            'Time',
            # 'LapTime',
            # 'PitOutTime',
            # 'PitInTime'
        ]

        # Convert all time columns to seconds
        for col in time_columns:
            if col in df.columns:
                self.convert_time_to_seconds(df, col)

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

        # Compute the set of pit stops for each driver
        # if not pit_stops:
        #     driver_pit_laps = {
        #     driver: set(df[(df['DriverNumber'] == driver) & (df['PitInTime_in_ms'].notna())]['LapNumber'])
        #     .union(df[(df['DriverNumber'] == driver) & (df['PitOutTime_in_ms'].notna())]['LapNumber'])
        #     for driver in df['DriverNumber'].unique()
        #     }
        #
        #     # Filter dataset using the precomputed mapping
        #     df = df[~df.apply(
        #         lambda row: row['LapNumber'] in driver_pit_laps[row['DriverNumber']],
        #         axis=1)]

        # One-hot encode 'Compound'
        if 'Compound' in df.columns:
            df = df[df['Compound'] != 'UNKNOWN'] # remove UNKNOWN compound records


        if 'IsAccurate' in df.columns and not include_not_accurate_and_pit:
            df = df[df['IsAccurate'] == True]

        # Map Team and Event
        df = self.map_features(df)

        # Columns to drop (irrelevant or redundant)
        drop_cols = [
            'Time_x', 'Driver', 'DriverNumber', 'Stint', 'PitOutTime', 'PitInTime',
            'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime',
            'Sector3SessionTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'IsPersonalBest',
            'FreshTyre', 'LapStartTime', 'LapStartDate', 'Deleted', 'DeletedReason', 'FastF1Generated',
            'IsAccurate', 'Compound_y', 'TyreLife_y', 'Time_y', 'Date', 'SessionTime', 'Source',
            'RelativeDistance', 'Status', 'Year', 'LapTime'
        ]

        # Drop irrelevant columns if they exist in the dataset
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Handle DistanceToDriverAhead and DriverAhead

        if 'Position' in df.columns:
            df.loc[df['Position'] == 1, 'DistanceToDriverAhead'] = 0
            df.loc[df['Position'] == 1, 'DriverAhead'] = 0

        df['DriverAhead'] = df['DriverAhead'].replace('', np.nan)
        df = df.dropna(subset=['DriverAhead'])
        df['TrackStatus'] = df['TrackStatus'].replace('', np.nan)
        df = df.dropna(subset=['TrackStatus'])

        df['DriverAhead'] = df['DriverAhead'].astype('int8')
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

    def preprocessing_and_normalization(self, include_not_accurate_and_pit, scaler_type="MinMaxScaler", normalize_by_driver=False, save_to_file=True):

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

        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        for year_folder in os.listdir(self.input_folder_path):
            year_folder_path = os.path.join(self.input_folder_path, year_folder)

            for file in os.listdir(year_folder_path):
                if not file.split('_')[1].startswith('Pre'):

                    year = int(file.split('_')[0])
                    event_name = file.split('_')[1].split('.')[0]

                    if self.given_event is not None and self.given_year is not None:
                        if self.given_event.lower() != event_name.lower() or self.given_year != year:
                            continue

                    print(f'Loading {event_name} of {year}...')

                    data_path = os.path.join(year_folder_path, file)
                    np_data = np.load(data_path, allow_pickle=True)['data']

                    print(f'Loaded! Converting to dataframe...')

                    df = pd.DataFrame(np_data, columns=all_columns)

                    print(f'Done!')

                    if self.given_event is not None and self.given_year is not None and self.given_driver is not None:
                        if event_name.lower() == self.given_event.lower() and year == self.given_year:

                            for driver in df['Driver'].unique():

                                # Filter the DataFrame for the current driver
                                if driver == self.given_driver:
                                    driver_df = df[df['Driver'] == driver].copy()

                                    # Normalize the data for the current driver
                                    cleaned_data = self.normalize_data(driver_df, scaler_type, include_not_accurate_and_pit=include_not_accurate_and_pit)

                                    print(f'Saving cleaned data for {driver}...')

                                    # Save the cleaned and normalized data for the current driver
                                    if save_to_file:
                                        np.savez_compressed(
                                            f'{self.output_folder_path}/test_data/selected_driver/{year}_{event_name}_{driver}_{scaler_type}_normalized_complete_wPits.npz',
                                            data=cleaned_data
                                        )
                                    else:
                                        return cleaned_data

                    elif normalize_by_driver:
                        for driver in df['Driver'].unique():

                            # Filter the DataFrame for the current driver
                            driver_df = df[df['Driver'] == driver].copy()

                            # Normalize the data for the current driver
                            cleaned_data = self.normalize_data(driver_df, scaler_type, include_not_accurate_and_pit=True)

                            print(f'Saving cleaned data for {driver}...')

                            # Save the cleaned and normalized data for the current driver
                            np.savez_compressed(
                                f'{self.output_folder_path}/test_data/normalized_data_by_driver/{year}_{event_name}_{driver}_{scaler_type}_normalized_complete_wPits.npz',
                                data=cleaned_data
                            )

                    else:
                        # Normalize the data for the entire event and year
                        failures_df = pd.read_csv('../Failures/old_19-24_all_events_anomalies.csv')
                        failures_of_year = failures_df[
                            (failures_df['Year'] == year) &
                            (failures_df['EventName'].str.replace(' ', '') == event_name)]

                        print(f'Found {failures_of_year.shape[0]} failures for {event_name} of {year}')
                        failure_found = False

                        if failures_of_year.shape[1] > 0:
                            driver_failed_list = list(failures_of_year['DriverNumber'].unique())
                            for failure in failures_of_year.iterrows():
                                driver_fail, event_fail, year_fail, problem_class = (failure[1]['DriverNumber'], failure[1]['EventName'], failure[1]['Year'], failure[1]['ProblemClass'])
                                event_fail = event_fail.replace(' ', '')
                                problem_class = problem_class.replace(' ', '')

                                if event_name.lower() == event_fail.lower() and year == year_fail:
                                    failure_found = True
                                    df['DriverNumber'] = df['DriverNumber'].astype(int)
                                    print(f'df: {df.shape}')

                                    df_failures = df[df['DriverNumber'] == driver_fail].copy()
                                    print(f'df_failures: {df_failures.shape}')
                                    if df_failures.shape[0] > 0:
                                        try:
                                            df_failures['Failures'] = problem_class
                                            cleaned_data_failures = self.normalize_data(df_failures, scaler_type, include_not_accurate_and_pit=True)
                                        except Exception as e:
                                            # Save the error to a text file
                                            error_message = f"{event_name}, {year}, failure({driver_fail}, {problem_class}): {str(e)}"
                                            with open("D:/error_log.txt", "a") as error_file:
                                                error_file.write(error_message + "\n")
                                            print("An error occurred. Details have been saved to 'error_log.txt'.")
                                            continue

                                        print(f'Saving failure data for {driver_fail} in {event_fail} of {year_fail}...')
                                        stime = time.time()
                                        np.savez_compressed(
                                            f'{output_folder}/train_data/train_data_only_failures/{year}_{event_name}_{scaler_type}_normalized_{driver_fail}_{problem_class}.npz',
                                            data=cleaned_data_failures)

                                        etime = time.time()
                                        print(f'Done in {etime - stime:.2f} seconds')

                            print(f'Saving {event_name} of {year}...')
                            stime = time.time()
                            print(f'df_without_failures before: {df.shape}')
                            df_without_failures = df[~df['DriverNumber'].isin(driver_failed_list)].copy()
                            print(f'df_without_failures after: {df_without_failures.shape}')
                            try:
                                cleaned_data_without_failures = self.normalize_data(df_without_failures, scaler_type, include_not_accurate_and_pit=include_not_accurate_and_pit)
                            except Exception as e:
                                # Save the error to a text file
                                error_message = f"{event_name}, {year}, train file without failures (removed): {str(e)}"
                                with open("D:/error_log.txt", "a") as error_file:
                                    error_file.write(error_message + "\n")
                                print("An error occurred. Details have been saved to 'error_log.txt'.")
                                continue
                            np.savez_compressed(
                                f'{output_folder}/train_data/train_data_without_failures/{year}_{event_name}_{scaler_type}_normalized.npz',
                                data=cleaned_data_without_failures)
                            etime = time.time()
                            print(f'Done in {etime - stime:.2f} seconds')

                        if not failure_found:
                            try:
                                cleaned_data = self.normalize_data(df, scaler_type, include_not_accurate_and_pit=include_not_accurate_and_pit)
                            except Exception as e:
                                # Save the error to a text file
                                error_message = f"{event_name}, {year}, train file without failures (absent): {str(e)}"
                                with open("D:/error_log.txt", "a") as error_file:
                                    error_file.write(error_message + "\n")
                                print("An error occurred. Details have been saved to 'error_log.txt'.")
                                continue
                            print(f'Saving {event_name} data of {year}...')
                            stime = time.time()
                            np.savez_compressed(
                                f'{output_folder}/train_data/train_data_without_failures/{year}_{event_name}_{scaler_type}_normalized.npz',
                                data=cleaned_data)
                            etime = time.time()
                            print(f'Done in {etime - stime:.2f} seconds')

if __name__ == '__main__':

    input_folder = 'D:/F1LLM_Datasets/npz_all_telemetry_data'   # EDIT THIS PATH
    output_folder = 'D:/F1LLM_Datasets/npz_normalized'  # EDIT THIS PATH

    if not os.path.exists(input_folder):
        print('Input folder does not exist. Exiting...')
        exit(1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    user_input = input('Select the operation to perform:\n '
                       '1. [TRAIN] Normalize ALL data by event and year REMOVING pit-stops, and driver with failures (saved separately for classification training)\n '
                       '2. [TRAIN] Normalize ALL data by event and year MAINTAINING pit-stops and REMOVING driver with failures (saved separately for classification training)\n '
                       '3. [TEST] Normalize data by driver, event and year MAINTAINING pit-stops\n '
                       '4. [TEST] Normalize a single file MAINTAINING pit-stops (driver, event, year)\n '
                       '--> ')

    if user_input == '1':

        if not os.path.exists(f'{output_folder}/train_data'):
            os.makedirs(f'{output_folder}/train_data')
        if not os.path.exists(f'{output_folder}/train_data/train_data_without_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_without_failures')
        if not os.path.exists(f'{output_folder}/train_data/train_data_only_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_only_failures')

        normalizer = Normalizer(input_folder, output_folder)
        normalizer.preprocessing_and_normalization(
            scaler_type="MinMaxScaler",
            normalize_by_driver=False,
            include_not_accurate_and_pit=False,
            save_to_file=True
        )

    elif user_input == '2':

        if not os.path.exists(f'{output_folder}/train_data'):
            os.makedirs(f'{output_folder}/train_data')
        if not os.path.exists(f'{output_folder}/train_data/train_data_without_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_without_failures')
        if not os.path.exists(f'{output_folder}/train_data/train_data_only_failures'):
            os.makedirs(f'{output_folder}/train_data/train_data_only_failures')

        normalizer = Normalizer(input_folder, output_folder)
        normalizer.preprocessing_and_normalization(
            scaler_type="MinMaxScaler",
            normalize_by_driver=False,
            include_not_accurate_and_pit=True,
            save_to_file=True
        )

    elif user_input == '3':

        if not os.path.exists(f'{output_folder}/test_data'):
            os.makedirs(f'{output_folder}/test_data')
        if not os.path.exists(f'{output_folder}/test_data/normalized_data_by_driver'):
            os.makedirs(f'{output_folder}/test_data/normalized_data_by_driver')

        normalizer = Normalizer(input_folder, output_folder)
        normalizer.preprocessing_and_normalization(normalize_by_driver=True)

    elif user_input == '4':

        driver = input('Enter the driver name (first 3 letters): ')
        driver = driver.upper()
        event = input('Enter the event name: ')
        event = event.replace(' ', '')
        year = input('Enter the year: ')

        if not os.path.exists(f'{output_folder}/test_data'):
            os.makedirs(f'{output_folder}/test_data')
        if not os.path.exists(f'{output_folder}/test_data/selected_driver'):
            os.makedirs(f'{output_folder}/test_data/selected_driver')

        normalizer = Normalizer(input_folder, output_folder, given_driver=driver, given_event=event, given_year=year)
        normalizer.preprocessing_and_normalization()
    else:
        print('Invalid input. Exiting...')
        exit(1)

