import fastf1 as ff1
import pandas as pd
from IPython.display import clear_output
import os
import numpy as np
import dask.dataframe as dd
from collections import defaultdict



# Function to merge laps with weather data (if taken)
def merge_laps_weather(laps, weather_data, include_weather):
    if not include_weather or weather_data is None:
        return laps

    # Select relevant weather columns
    weather_data = weather_data[
        ['Time', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']]

    # Perform a backward merge to align weather data with laps
    return pd.merge_asof(
        laps.sort_values(by="Time"),
        weather_data.sort_values(by="Time"),
        on="Time",
        direction="backward"
    )


# Function to extract telemetry data for the last 3 laps
def get_telemetry_data(target_laps, team, year, event_name):
    telemetry_frames = []

    for _, lap in target_laps.iterrows():

        telemetry = lap.get_telemetry()

        if telemetry is not None:
            # Select relevant telemetry columns
            # telemetry = telemetry[['Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake',
            #                        'DRS', 'Distance', 'X', 'Y', 'Z']]

            telemetry['Distance'] = telemetry['Distance'].clip(lower=0)  # Set negative distance values to 0
            telemetry['LapNumber'] = lap['LapNumber']
            telemetry['Team'] = team
            telemetry['Year'] = f"{year}"
            telemetry['Event'] = f"{event_name}"
            telemetry_frames.append(telemetry)

    # Combine telemetry frames if available
    return pd.concat(telemetry_frames) if telemetry_frames else None


# Function to clean and format the final data
def preprocessing(final_data):
    # Drop redundant 'Time_x' column (returned from weather) if it exists
    if 'Time_x' in final_data.columns:
        final_data.drop(columns=['Time_x'], inplace=True)

    # Format 'Time_y' as mm:ss.sss
    final_data['Time_y'] = final_data['Time_y'].apply(
        lambda x: f"{int(x.total_seconds() // 60):02}:{int(x.total_seconds() % 60):02}.{int(x.microseconds / 1000):03}"
    )
    final_data.rename(columns={'Time_y': 'TimeXY'}, inplace=True)

    final_data["DriverNumber"] = final_data["DriverNumber"].astype(str)
    final_data['LapTime'] = pd.to_timedelta(final_data['LapTime'], errors='coerce')
    final_data['LapNumber'] = final_data['LapNumber'].astype(int)
    final_data['Stint'] = pd.to_numeric(final_data['Stint'], errors='coerce')
    final_data['Stint'] = final_data['Stint'].fillna(0).astype(int)
    final_data['PitOutTime'] = pd.to_timedelta(final_data['PitOutTime'], errors='coerce')
    final_data['PitInTime'] = pd.to_timedelta(final_data['PitInTime'], errors='coerce')
    final_data['Sector1Time'] = pd.to_timedelta(final_data['Sector1Time'], errors='coerce')
    final_data['Sector2Time'] = pd.to_timedelta(final_data['Sector2Time'], errors='coerce')
    final_data['Sector3Time'] = pd.to_timedelta(final_data['Sector3Time'], errors='coerce')
    final_data['Sector1SessionTime'] = pd.to_timedelta(final_data['Sector1SessionTime'], errors='coerce')
    final_data['Sector2SessionTime'] = pd.to_timedelta(final_data['Sector2SessionTime'], errors='coerce')
    final_data['Sector3SessionTime'] = pd.to_timedelta(final_data['Sector3SessionTime'], errors='coerce')
    final_data["SpeedI1"] = final_data["SpeedI1"].astype(float) if final_data["SpeedI1"] is not None else 0
    final_data["SpeedI2"] = final_data["SpeedI2"].astype(float)
    final_data["SpeedFL"] = final_data["SpeedFL"].astype(float)
    final_data["SpeedST"] = final_data["SpeedST"].astype(float)
    final_data['IsPersonalBest'] = final_data['IsPersonalBest'].astype(str).str.upper() == 'TRUE'
    final_data['Compound_x'] = final_data['Compound_x'].astype(str)
    final_data['Compound_y'] = final_data['Compound_y'].astype(str)
    final_data['TyreLife_x'] = pd.to_numeric(final_data['TyreLife_x'], errors='coerce')
    final_data['TyreLife_x'] = final_data['TyreLife_x'].fillna(0).astype(int)
    final_data['TyreLife_y'] = pd.to_numeric(final_data['TyreLife_y'], errors='coerce')
    final_data['TyreLife_y'] = final_data['TyreLife_y'].fillna(0).astype(int)
    final_data['FreshTyre'] = final_data['FreshTyre'].astype(str).str.upper() == 'TRUE'
    final_data['Team'] = final_data['Team'].astype(str)
    final_data['LapStartTime'] = pd.to_timedelta(final_data['LapStartTime'], errors='coerce')
    final_data['LapStartDate'] = pd.to_datetime(final_data['LapStartDate'], format="%m/%d/%Y %I:%M:%S %p").dt.time

    final_data['TrackStatus'] = pd.to_numeric(final_data['TrackStatus'], errors='coerce')
    final_data['TrackStatus'] = final_data['TrackStatus'].fillna(0).astype(int)

    final_data['Position'] = pd.to_numeric(final_data['Position'], errors='coerce')
    final_data["Position"] = final_data["Position"].fillna(0).astype(int)

    final_data['Deleted'] = final_data['Deleted'].astype(str).str.upper() == 'TRUE'
    final_data['DeletedReason'] = final_data['DeletedReason'].astype(str)
    final_data['FastF1Generated'] = final_data['FastF1Generated'].astype(str).str.upper() == 'TRUE'
    final_data['IsAccurate'] = final_data['IsAccurate'].astype(str).str.upper() == 'TRUE'

    final_data['TimeXY'] = pd.to_timedelta(final_data['TimeXY'], errors='coerce')
    final_data["AirTemp"] = final_data["AirTemp"].astype(float)
    final_data["Humidity"] = final_data["Humidity"].astype(float)
    final_data["Pressure"] = final_data["Pressure"].astype(float)
    final_data["Rainfall"] = final_data["Rainfall"].astype(str).str.upper() == 'TRUE'
    final_data['TrackTemp'] = final_data['TrackTemp'].astype(float)
    final_data["WindDirection"] = final_data["WindDirection"].astype(float)
    final_data["WindSpeed"] = final_data["WindSpeed"].astype(float)

    final_data['Date'] = pd.to_datetime(final_data['Date'], format="%m/%d/%Y %I:%M:%S %p").dt.time
    final_data['SessionTime'] = pd.to_timedelta(final_data['SessionTime'], errors='coerce')
    final_data['DriverAhead'] = final_data['DriverAhead'].astype(str)
    final_data['DistanceToDriverAhead'] = final_data['DistanceToDriverAhead'].astype(float)
    final_data['Time'] = pd.to_timedelta(final_data['Time'], errors='coerce')
    final_data['RPM'] = final_data['RPM'].astype(int)
    final_data['Speed'] = final_data['Speed'].astype(int)
    final_data = final_data[final_data['Speed'] > 0]
    final_data['nGear'] = final_data['nGear'].astype(int)
    final_data['Throttle'] = final_data['Throttle'].astype(int)
    final_data['Brake'] = final_data['Brake'].astype(str).str.upper() == 'TRUE'
    final_data['DRS'] = final_data['DRS'].astype(int)
    final_data['Source'] = final_data['Source'].astype(str)
    final_data["Distance"] = final_data["Distance"].astype(float)
    final_data['RelativeDistance'] = final_data['RelativeDistance'].astype(float)
    final_data['Status'] = final_data['Status'].astype(str)
    final_data['X'] = final_data['X'].astype(int)
    final_data['Y'] = final_data['Y'].astype(int)
    final_data['Z'] = final_data['Z'].astype(int)
    final_data['Year'] = final_data['Year'].astype(int)
    final_data['Event'] = final_data['Event'].astype(str)

    return final_data


# Function to save the final processed data
def save_data(final_data, output_folder, year, event_name):
    # Create the main folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create the year folder inside the main folder
    year_folder = os.path.join(output_folder, str(year))
    os.makedirs(year_folder, exist_ok=True)

    # Replace spaces in event name with underscores
    event_name = event_name.replace(' ', '')

    # Save the final CSV file in the year folder
    output_file = os.path.join(year_folder, f"{event_name}_{year}.csv")
    final_data.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


def all_drivers_data_from_races(output_folder, include_weather=True, save_file=True, year=2018):
    # Enable FastF1 cache
    ff1.Cache.enable_cache('../cache')

    # Retrieve the schedule for the specified year
    schedule = ff1.get_event_schedule(year)
    schedule = schedule[1:2]  # Limiting to one race for testing; modify as needed

    if schedule is None or schedule.empty:
        print("No data to process for the specified year.")
        return

    # Iterate over each race in the schedule
    for _, race_info in schedule.iterrows():
        event_name = race_info['EventName']

        try:
            # Load session data for the race
            session = ff1.get_session(year, event_name, 'Q')
            session.load(laps=True, telemetry=True, weather=include_weather)
        except Exception as e:
            print(f"Error loading session for {event_name} ({year}): {e}")
            continue

        # Initialize DataFrames for laps and telemetry
        all_teams_laps = []
        all_telemetry = []

        # Loop through all drivers in the session
        teams = session.laps["Team"].unique()
        for team in teams:
            try:
                # Get laps for the current driver
                team_laps = session.laps.pick_team(team)
                quick_laps = team_laps.pick_quicklaps()
                if quick_laps.empty:
                    print(f"No laps data available for driver {team} in {event_name}.")
                    continue
                all_teams_laps.append(quick_laps)

                # Extract telemetry data
                telemetry_data = get_telemetry_data(quick_laps, team, year, event_name)
                if telemetry_data is not None:
                    all_telemetry.append(telemetry_data)
            except Exception as e:
                print(f"Error processing {team} in {event_name}: {e}")
                continue

        # Combine laps and telemetry data for all drivers
        if not all_teams_laps or not all_telemetry:
            print(f"No valid data for {event_name}.")
            continue

        all_laps_df = pd.concat(all_teams_laps, ignore_index=True)
        all_telemetry_df = pd.concat(all_telemetry, ignore_index=True)
        session_laps_df = pd.DataFrame(session.laps.pick_quicklaps())

        laps_data = all_laps_df[['LapNumber', 'Compound', 'TyreLife', 'Time']]

        # Merge laps with weather data if included
        if include_weather:
            laps_with_weather = merge_laps_weather(laps_data, session.weather_data, include_weather)
        else:
            laps_with_weather = laps_data

        # Merge session laps with laps+weather
        try:
            laps_with_weather = pd.merge_asof(
                session_laps_df.sort_values(by="LapNumber"),
                laps_with_weather.sort_values(by="LapNumber"),
                on="LapNumber",
                # by="Time",
                direction="backward"
            )
        except Exception as e:
            print(session_laps_df.columns, laps_with_weather.columns)
            print(f"Error merging laps with weather for {event_name}: {e}")
            continue

        # Merge laps+weather with telemetry data
        try:
            final_data = pd.merge(
                laps_with_weather,
                all_telemetry_df,
                on=["Team", "LapNumber"],
                how="inner",
                suffixes=("_laps", "_telemetry")  # Specifica suffissi personalizzati
            )
        except Exception as e:
            print(laps_with_weather.columns, all_telemetry_df.columns)
            print(f"Error merging laps+weather with telemetry for {event_name}: {e}")
            continue

        # Clean and format the final data
        final_data = preprocessing(final_data)

        # Save the final data if requested
        if save_file:
            save_data(final_data, output_folder, year, event_name)

        # Clear the output after processing each race
        clear_output()

    print("All CSV files have been saved.")

import os
import dask.dataframe as dd
import numpy as np
import pandas as pd


def convert_csv_to_npz(output_folder, input_folder='AllTelemetryData'):
    os.makedirs(output_folder, exist_ok=True)

    # Define the dtype for each column
    dtype_dict = {
        "DriverNumber": str,
        "LapNumber": int,
        "Stint": int,
        "SpeedI1": float,
        "SpeedI2": float,
        "SpeedFL": float,
        "SpeedST": float,
        "IsPersonalBest": str,
        "Compound_x": str,
        "Compound_y": str,
        "TyreLife_x": int,
        "TyreLife_y": int,
        "FreshTyre": bool,
        "Team": str,
        "TrackStatus": int,
        "Position": int,
        "Deleted": bool,
        "DeletedReason": str,
        "FastF1Generated": bool,
        "IsAccurate": bool,
        "AirTemp": float,
        "Humidity": float,
        "Pressure": float,
        "Rainfall": bool,
        "TrackTemp": float,
        "WindDirection": float,
        "WindSpeed": float,
        "DriverAhead": str,
        "DistanceToDriverAhead": float,
        "RPM": int,
        "Speed": int,
        "nGear": int,
        "Throttle": int,
        "Brake": bool,
        "DRS": int,
        "Source": str,
        "Distance": float,
        "RelativeDistance": float,
        "Status": str,
        "X": int,
        "Y": int,
        "Z": int,
        "Year": int,
        "Event": str,
        # Specify time columns as object
        "PitInTime": object,
        "PitOutTime": object,
        "Sector1SessionTime": object,
        "Sector1Time": object,
        "Sector2SessionTime": object,
        "Sector2Time": object,
        "Sector3SessionTime": object,
        "Sector3Time": object,
        "LapStartTime": object,
        "SessionTime": object,
        "LapTime": object,
        "TimeXY": object,
        "LapStartDate": object,
        "Date": object,
    }

    # Define time columns to convert
    time_columns = [
        'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
        'LapStartTime', 'SessionTime', 'TimeXY', 'LapStartDate', 'Date',
        'PitInTime', 'PitOutTime'
    ]

    # Loop through each year folder in the input folder
    for year_folder in os.listdir(input_folder):
        year_folder_path = os.path.join(input_folder, year_folder)

        if os.path.isdir(year_folder_path):
            print(f"Processing year folder: {year_folder}")

            # Loop through each file in the current year folder
            for file in os.listdir(year_folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(year_folder_path, file)
                    print(f'Loading data for {file}')

                    # Use Pandas to read the CSV file with specified dtype
                    data = pd.read_csv(file_path, dtype=dtype_dict)

                    # Convert time columns to milliseconds
                    for col in time_columns:
                        if col in data.columns:
                            try:
                                # Convert to timedelta and get total seconds in milliseconds
                                data[col] = pd.to_timedelta(data[col], errors='coerce').dt.total_seconds() * 1000
                            except Exception as e:
                                print(f"Error processing column {col}: {e}")

                    # Convert the DataFrame to numpy array
                    data_np = data.to_numpy()

                    # Define the output path for the .npz file
                    output_file_path = os.path.join(output_folder, f"{year_folder}_{os.path.splitext(file)[0]}.npy")

                    # Save the dataframe as a compressed numpy file
                    np.save(output_file_path, data=data_np)
                    print(f'Saved {file} as {output_file_path}')

    print(f"All files have been processed and saved to: {output_folder}")

    print(f"All files have been processed and saved to: {output_folder}")


def merge_npz_by_year(input_folder, output_folder, chunk_size=1_000_000):
    """
    Merge .npz files by year into a single .npz file per year, using chunk processing.

    Parameters:
    - input_folder: Folder containing the .npz files.
    - output_folder: Folder to save the merged .npz files.
    - chunk_size: Number of rows to process at a time to avoid memory issues.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to group files by year
    files_by_year = defaultdict(list)

    # List all .npz files in the input folder
    npz_files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]

    # Group files by year
    for file in npz_files:
        try:
            # Extract the year from the filename (assuming the year is the first part of the filename)
            year = file.split('_')[1]
            files_by_year[year].append(file)
        except Exception as e:
            print(f"Error processing filename {file}: {e}")

    print(f"Found files for years: {list(files_by_year.keys())}")

    # Process each year
    for year, files in files_by_year.items():
        print(f"Merging files for year {year}: {files}")

        data_accumulator = []  # List to collect chunks

        for file in files:
            file_path = os.path.join(input_folder, file)
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    for key in data.files:
                        array = data[key]
                        print(f"Processing file: {file}, key: {key}, shape: {array.shape}")

                        # Process array in chunks
                        for i in range(0, array.shape[0], chunk_size):
                            chunk = array[i:i + chunk_size]
                            data_accumulator.append(chunk)

            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Concatenate all chunks for the year
        if data_accumulator:
            try:
                concatenated_data = np.concatenate(data_accumulator, axis=0)
                output_file = os.path.join(output_folder, f"merged_{year}.npz")
                np.savez_compressed(output_file, data=concatenated_data)
                print(f"Saved merged data for year {year} with shape: {concatenated_data.shape}")
            except Exception as e:
                print(f"Error saving merged data for year {year}: {e}")
        else:
            print(f"No data to merge for year {year}.")


input_folder = 'C:/Users/rioti/Documents/GitHub/2024-P3-F1LLM/AllTelemetryData'
output_folder = 'C:/2024-P3-F1LLM/QualyData'
for year in range(2021, 2022):
   all_drivers_data_from_races(
       output_folder,
       include_weather=True,
       save_file=True,
       year=year,
   )

# merge_npz_by_year(input_folder, output_folder)
