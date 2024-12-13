import fastf1 as ff1
import pandas as pd
import os
from IPython.display import clear_output
import all_race_data_extraction


# Function to get the last 3 laps of the given driver
def get_last_3_laps(laps):
    # Determine the last lap number and filter the last 3 laps
    last_lap_number = laps['LapNumber'].max()
    start_lap_number = max(1, last_lap_number - 2)
    return laps[laps['LapNumber'].between(start_lap_number, last_lap_number)]


# Main function to process data for all races
def data_from_races(db_path, output_folder, include_weather=True, save_file=True, years_to_include=None):
    # Enable FastF1 cache
    ff1.Cache.enable_cache('cache')

    # Load the CSV file containing race data
    df = pd.read_csv(db_path)

    # Filter the data for specific years if provided
    if years_to_include is not None:
        df = df[df['Year'].isin(years_to_include)]

    if df.empty:
        print("No data to process for the specified years.")
        return

    # Iterate over each driver in the race data
    for _, row in df.iterrows():
        year = row['Year']
        event_name = row['EventName']
        driver_number = row['DriverNumber']

        # Load session data
        session, laps = all_race_data_extraction.load_session(year, event_name, driver_number, include_weather)
        if session is None or laps is None:
            continue

        # Extract the last 3 laps of the driver
        target_laps = get_last_3_laps(laps)
        laps_data = target_laps[['DriverNumber', 'LapNumber', 'Compound', 'TyreLife', 'Time']]

        # Merge laps with weather data if included
        if include_weather:
            laps_with_weather = all_race_data_extraction.merge_laps_weather(laps_data, session.weather_data, include_weather)
        else:
            laps_with_weather = laps_data

        # Extract telemetry data
        telemetry_data = all_race_data_extraction.get_telemetry_data(target_laps, driver_number, year, event_name)
        if telemetry_data is None:
            print(f"No telemetry data for driver {driver_number} in {event_name}.")
            continue

        # Merge laps+weather with telemetry data
        laps_with_weather['DriverNumber'] = laps_with_weather['DriverNumber'].astype(str)
        telemetry_data['DriverNumber'] = telemetry_data['DriverNumber'].astype(str)

        final_data = pd.merge(
            laps_with_weather,
            telemetry_data,
            on=['DriverNumber', 'LapNumber'],
            how='inner'
        )

        # Clean and format the final data
        final_data = all_race_data_extraction.preprocessing(final_data)

        # Save the final data if requested
        if save_file:
            all_race_data_extraction.save_data(final_data, output_folder, year, event_name, driver_number)

        clear_output()

    print("All csv have been saved")


# Input and output paths
db_path = 'Failures2014_2024_cleaned.csv'
output_folder = 'TelemetryData'

# Execute the main function
data_from_races(
    db_path,
    output_folder,
    include_weather=True,
    save_file=True,
    years_to_include=range(2023, 2024)  # List of years to include
)
