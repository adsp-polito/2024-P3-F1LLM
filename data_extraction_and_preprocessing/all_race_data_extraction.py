import fastf1 as ff1
import os
import numpy as np
import pandas as pd

# Merge laps with weather data
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

# Retrieve telemetry data
def get_telemetry_data(target_laps, driver_number, year, event_name):
    telemetry_frames = []

    for _, lap in target_laps.iterrows():

        telemetry = lap.get_telemetry()

        if telemetry is not None:

            # telemetry['Distance'] = telemetry['Distance'].clip(lower=0)  # Set negative distance values to 0
            telemetry['LapNumber'] = lap['LapNumber']
            telemetry['DriverNumber'] = driver_number
            telemetry['Year'] = f"{year}"
            telemetry['Event'] = f"{event_name}"
            telemetry_frames.append(telemetry)

    # Combine telemetry frames if available
    return pd.concat(telemetry_frames) if telemetry_frames else None

# Save the data
def save_data(final_data, output_folder_path, year, event_name, npz_instead_csv):

    # Create the main folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Create the year folder inside the main folder
    year_folder = os.path.join(output_folder_path, str(year))
    os.makedirs(year_folder, exist_ok=True)

    # Replace spaces in event name with underscores
    event_name = event_name.replace(' ', '')

    # Save the final npz or CSV file in the year folder
    if npz_instead_csv:
        output_file = os.path.join(year_folder, f"{year}_{event_name}.npz")
        final_data = final_data.to_numpy()
        np.savez_compressed(output_file, data=final_data)
    else:
        output_file = os.path.join(year_folder, f"{year}_{event_name}.csv")
        final_data.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

# Main function
def all_drivers_data_from_races(output_folder_path, include_weather=True, save_file=True, year=2024, npz_instead_csv=True):

    # Enable FastF1 cache
    ff1.Cache.enable_cache('../cache')

    # Retrieve the schedule for the specified year
    schedule = ff1.get_event_schedule(year)
    # schedule = schedule[1:2]  # Limiting to one race for testing; modify as needed

    if schedule is None or schedule.empty:
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
                driver_laps = session.laps.pick_drivers(driver)
                if driver_laps.empty:
                    print(f"No laps data available for driver {driver} in {event_name}.")
                    continue
                all_driver_laps.append(driver_laps)

                # Extract telemetry data
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
        session_laps_df = pd.DataFrame(session.laps)

        laps_data = all_laps_df[['DriverNumber', 'LapNumber', 'Compound', 'TyreLife', 'Time']]

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
                by="DriverNumber",
                direction="backward"
            )
        except Exception as e:
            print(f"Error merging laps with weather for {event_name}: {e}")
            continue

        # Merge laps+weather with telemetry data
        try:
            final_data_df = pd.merge(
                laps_with_weather,
                all_telemetry_df,
                on=["DriverNumber", "LapNumber"],
                how="inner",
                suffixes=("_laps", "_telemetry")
            )
        except Exception as e:
            print(f"Error merging laps+weather with telemetry for {event_name}: {e}")
            continue

        # Save the final data if requested
        if save_file:
            save_data(final_data_df, output_folder_path, year, event_name, npz_instead_csv=npz_instead_csv)


    print("All files have been saved.")


output_folder = '../test_outputs'
for year in range(2024, 2025):
   all_drivers_data_from_races(
       output_folder,
       include_weather=True,
       save_file=True,
       year=year,
       npz_instead_csv=True
   )