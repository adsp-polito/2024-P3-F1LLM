import fastf1 as ff1
import pandas as pd
import os
from IPython.display import clear_output


# Function to load and prepare the session data
def load_session(year, event_name, driver_number, include_weather):
    try:
        # Retrieve and Load the session
        session = ff1.get_session(year, event_name, 'R')
        session.load(laps=True, telemetry=True, weather=include_weather)
        
        # Filter laps for the given driver
        laps = session.laps.pick_drivers(driver_number)
        
        # Check if data are returned
        if laps.empty:
            print(f"No data available for driver {driver_number} in {event_name}.")
            return None, None
        
        return session, laps
    
    except Exception as e:
        print(f"Error loading data for {year} {event_name}: {e}")
        return None, None


# Function to get the last 3 laps of the given driver
def get_last_3_laps(laps):
    
    # Determine the last lap number and filter the last 3 laps
    last_lap_number = laps['LapNumber'].max()
    start_lap_number = max(1, last_lap_number - 2)
    return laps[laps['LapNumber'].between(start_lap_number, last_lap_number)]


# Function to merge laps with weather data (if taken)
def merge_laps_weather(laps, weather_data, include_weather):
    if not include_weather or weather_data is None:
        return laps
    
    # Select relevant weather columns
    weather_data = weather_data[['Time', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']]
    
    # Perform a backward merge to align weather data with laps
    return pd.merge_asof(
        laps.sort_values(by="Time"),
        weather_data.sort_values(by="Time"),
        on="Time",
        direction="backward"
    )


# Function to extract telemetry data for the last 3 laps
def get_telemetry_data(target_laps, driver_number, year, event_name):
    telemetry_frames = []

    for _, lap in target_laps.iterrows():
        try:
            # Directly fetch telemetry without relying on position data
            telemetry = lap.get_car_data()  # This fetches telemetry without needing pos_data

            if telemetry is not None:
                # Select relevant telemetry columns
                telemetry = telemetry[['Time', 'RPM', 'Speed', 'Throttle', 'Brake', 'DRS']]

                # Add metadata columns
                telemetry['LapNumber'] = lap['LapNumber']
                telemetry['DriverNumber'] = driver_number
                telemetry['Year'] = f"{year}"
                telemetry['Event'] = f"{event_name}"
                telemetry_frames.append(telemetry)
        except Exception as e:
            print(
                f"Error processing telemetry for Driver {driver_number}, Lap {lap['LapNumber']}, Event {event_name} ({year}): {e}")

    return pd.concat(telemetry_frames) if telemetry_frames else None


# Function to clean and format the final data
def preprocessing(final_data):
    # Drop redundant 'Time_x' column (returned from weather) if it exists
    if 'Time_x' in final_data.columns:
        final_data.drop(columns=['Time_x'], inplace=True)

    # Format 'Time_y' as mm:ss.sss
    if 'Time_y' in final_data.columns:
        final_data['Time_y'] = final_data['Time_y'].apply(
            lambda
                x: f"{int(x.total_seconds() // 60):02}:{int(x.total_seconds() % 60):02}.{int(x.microseconds / 1000):03}"
        )
        final_data.rename(columns={'Time_y': 'Time'}, inplace=True)

    # Add 'Distance' column if it doesn't exist
    if 'Distance' not in final_data.columns:
        final_data['Distance'] = None
    else:
        # Round 'Distance' to 3 decimal places
        final_data['Distance'] = final_data['Distance'].round(3)

    # Remove rows where speed is 0
    if 'Speed' in final_data.columns:
        final_data = final_data[final_data['Speed'] > 0]

    # Convert LapNumber to int if it exists
    if 'LapNumber' in final_data.columns:
        final_data['LapNumber'] = final_data['LapNumber'].astype(int)

    return final_data


# Function to save the final processed data
def save_data(final_data, output_folder, year, event_name, driver_number):
    
    # Create the main folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the year folder inside the main folder
    year_folder = os.path.join(output_folder, str(year))
    os.makedirs(year_folder, exist_ok=True)

    # Save the final CSV file in the year folder
    output_file = os.path.join(year_folder, f"{driver_number}_{event_name}_{year}.csv")
    final_data.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


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
        session, laps = load_session(year, event_name, driver_number, include_weather)
        if session is None or laps is None:
            continue

        # Extract the last 3 laps of the driver
        target_laps = get_last_3_laps(laps)
        laps_data = target_laps[['DriverNumber', 'LapNumber', 'Compound', 'TyreLife', 'Time']]

        # Merge laps with weather data if included
        if include_weather:
            laps_with_weather = merge_laps_weather(laps_data, session.weather_data, include_weather)
        else:
            laps_with_weather = laps_data

        # Extract telemetry data
        telemetry_data = get_telemetry_data(target_laps, driver_number, year, event_name)
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
        final_data = preprocessing(final_data)

        # Save the final data if requested
        if save_file:
            save_data(final_data, output_folder, year, event_name, driver_number)
        
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
    years_to_include=range(2018, 2019)  # List of years to include
)
