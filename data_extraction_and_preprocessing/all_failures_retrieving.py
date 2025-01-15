import fastf1 as ff1
import pandas as pd

all_data = []

year = range(2019, 2025)
for y in year:
    try:
        schedule = ff1.get_event_schedule(y)
        for race in schedule.iterrows():
            race_info = race[1]
            event_name = race_info['EventName']  # Nome specifico della gara
            print(f"Loading data for {event_name} ({race_info['EventDate']})")

            try:
                race_session = ff1.get_session(y, event_name, 'R')
                race_session.load(telemetry=False, laps=False, weather=False)  # IMPORTANT
                print(f"Loaded data for {event_name} ({race_info['EventDate']})")

                # Filtra per status rilevanti
                relevant_status = race_session.results[(race_session.results['Status'] != 'Finished') &
                                                       (~race_session.results['Status'].str.contains('lap', case=False,
                                                                                                     na=False)) &
                                                       (race_session.results['Status'] != 'Collision') &
                                                       (race_session.results['Status'] != 'Disqualified') &
                                                       (race_session.results['Status'] != 'Collision damage') &
                                                       (race_session.results['Status'] != 'Wheel nut') &
                                                       (race_session.results['Status'] != 'Accident') &
                                                       (race_session.results['Status'] != 'Retired') &
                                                       (race_session.results['Status'] != 'Withdrew') &
                                                       (race_session.results['Status'] != 'Spun off') &
                                                       (race_session.results['Status'] != 'Seat') &
                                                       (race_session.results['Status'] != 'Debris') &
                                                       (race_session.results['Status'] != 'Excluded') &
                                                       (race_session.results['Status'] != 'Illness')]

                # Crea il DataFrame con solo dati rilevanti per le anomalies
                partial_csv = relevant_status[['DriverNumber', 'Status']].reset_index(drop=True)
                event_name = race_session.event['EventName']
                event_date = race_session.event['EventDate']
                partial_csv = partial_csv.assign(EventName=event_name, EventDate=event_date)

                all_data.append(partial_csv)

            except Exception as e:
                print(f"Failed to load data for {event_name}: {e}")
    except Exception as e:
        print(f"Failed to process year {year}: {e}")

print('All data loaded!')

#concat the two dataframes
anomalies_df = pd.concat(all_data, ignore_index=True)

anomalies_df['Year'] = anomalies_df['EventDate'].apply(lambda x: x.year)
anomalies_df.drop(columns=['EventDate'], inplace=True)

categories = {
    "Aerodynamics and Tyres": ["Undertray", "Rear wing", "Front wing", "Damage", "Puncture", "Wheel", "Tyre"],
    "Engine": ["Engine"],
    "Power Unit": ["Power Unit", "Power loss", "Turbo"],
    "Cooling System": ["Overheating", "Water pressure", "Water leak", "Cooling system", "Radiator", "Water pump"],
    "Suspension and Drive": ["Suspension", "Steering", "Driveshaft", "Differential", "Vibrations", "Hydraulics"],
    "Braking System": ["Brakes"],
    "Transmission and Gearbox": ["Gearbox", "Transmission"],
    "Others": ["Mechanical", "Exhaust", "Oil leak", "Technical", "Out of fuel", "Fuel pump", "Fuel pressure", "Fuel leak", "Electronics", "Electrical", "Battery"],
}

# Reverse the mapping for easy lookup
status_to_class = {status: category for category, statuses in categories.items() for status in statuses}

# Add the new column
anomalies_df['ProblemClass'] = anomalies_df['Status'].map(status_to_class)

# Reorder the columns
anomalies_df = anomalies_df[['DriverNumber', 'EventName', 'Year', 'Status', 'ProblemClass']]

anomalies_df.to_csv("../Dataset/19-24_all_events_anomalies.csv", index=False)
