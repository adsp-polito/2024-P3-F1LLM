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
        
        telemetry = lap.get_telemetry()
        
        if telemetry is not None:
            
            # Select relevant telemetry columns
            telemetry = telemetry[['Time', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake',
                                   'DRS', 'Distance', 'X', 'Y', 'Z']]
            
            telemetry['Distance'] = telemetry['Distance'].clip(lower=0) # Set negative distance values to 0
            telemetry['LapNumber'] = lap['LapNumber']
            telemetry['DriverNumber'] = driver_number
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
    final_data.rename(columns={'Time_y': 'Time'}, inplace=True)
    
    # Remove rows where speed is 0
    final_data = final_data[final_data['Speed'] > 0]
    
    # Convert LapNumber to int
    final_data['LapNumber'] = final_data['LapNumber'].astype(int)
    
    # Round Distance to 3 decimal places
    final_data['Distance'] = final_data['Distance'].round(3)
    
    return final_data

# Function to save the final processed data
def save_data(final_data, output_folder, year, event_name, driver_number):
    
    # Create the main folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the year folder inside the main folder
    year_folder = os.path.join(output_folder, str(year))
    os.makedirs(year_folder, exist_ok=True)
    
    # Replace spaces in event name with underscores
    event_name = event_name.replace(' ', '')

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
    
def all_drivers_data_from_races(output_folder, include_weather=True, save_file=True, year=2018):
    import fastf1 as ff1
    import pandas as pd
    import os
    from IPython.display import clear_output

    # Enable FastF1 cache
    ff1.Cache.enable_cache('cache')

    # Retrieve the schedule for the specified year
    schedule = ff1.get_event_schedule(year)
    # schedule = schedule[1:2]

    if schedule is None:
        print("No data to process for the specified year.")
        return

    # Iterate over each race in the schedule
    for _, race_info in schedule.iterrows():
        event_name = race_info['EventName']

        try:
            # Load session data for the race
            session = ff1.get_session(year, event_name, 'R')
            session.load(laps=True, telemetry=True, weather=include_weather)
        except Exception as e:
            print(f"Error loading session for {event_name} ({year}): {e}")
            continue

        # Initialize DataFrames for laps and telemetry
        all_driver_laps = []
        all_telemetry = []

        # Loop through all drivers in the session
        for driver in session.drivers:
            try:
                # Get laps for the current driver
                driver_laps = session.laps.pick_driver(driver)  # Already a pandas DataFrame
                if driver_laps.empty:
                    print(f"No laps data available for driver {driver} in {event_name}.")
                    continue
                all_driver_laps.append(driver_laps)

                # Extract telemetry data for the last 3 laps of the driver
                # last_3_laps = get_last_3_laps(driver_laps)
                telemetry_data = get_telemetry_data(driver_laps, driver, year, event_name)

                if telemetry_data is not None:
                    all_telemetry.append(telemetry_data)
            except Exception as e:
                print(f"Error processing driver {driver} in {event_name}: {e}")
                continue

        # Combine laps and telemetry data for all drivers
        if not all_driver_laps or not all_telemetry:
            print(f"No valid data for {event_name}.")
            continue

        all_laps_df = pd.concat(all_driver_laps, ignore_index=True)
        all_telemetry_df = pd.concat(all_telemetry, ignore_index=True)

        # Select relevant columns for laps
        laps_data = all_laps_df[['DriverNumber', 'LapNumber', 'Compound', 'TyreLife', 'Time']]

        # Merge laps with weather data if included
        if include_weather:
            laps_with_weather = merge_laps_weather(laps_data, session.weather_data, include_weather)
        else:
            laps_with_weather = laps_data

        # Merge laps+weather with telemetry data
        laps_with_weather['DriverNumber'] = laps_with_weather['DriverNumber'].astype(str)
        all_telemetry_df['DriverNumber'] = all_telemetry_df['DriverNumber'].astype(str)

        final_data = pd.merge(
            laps_with_weather,
            all_telemetry_df,
            on=['DriverNumber', 'LapNumber'],
            how='inner'
        )

        # Clean and format the final data
        final_data = preprocessing(final_data)

        # Save the final data if requested
        if save_file:
            save_data(final_data, output_folder, year, event_name, "all_drivers")
        
        # Clear the output after processing each race
        clear_output()

    print("All CSV files have been saved.")

# Input and output paths
db_path = 'Failures2014_2024_cleaned.csv'
output_folder = 'TelemetryData'
print("1 -> failures, 2 -> all drivers")
input = int(input("Enter your choice: "))

if input == 1:
# Execute the main function
    data_from_races(
        db_path,
        output_folder,
        include_weather=True,
        save_file=True,
        years_to_include=range(2023, 2024)  # List of years to include
    )
else:
    output_folder2 = 'AllTelemetryData'
    all_drivers_data_from_races(
        output_folder2,
        include_weather=True,
        save_file=True,
        year=2019
    )
