import pandas as pd

# Load your CSV
df = pd.read_csv("cleaned_chicago_crime_data.csv")

# Drop rows with any null or empty string values
df = df.dropna()
df = df[~df.apply(lambda x: x.astype(str).str.strip().eq('').any(), axis=1)]

# Parse date column for time-based features
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['hour_of_day'] = df['Date'].dt.hour
df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

# Assign time slots based on hour_of_day
df['time_slot_morning'] = df['hour_of_day'].apply(lambda x: 1 if 6 <= x < 12 else 0)
df['time_slot_afternoon'] = df['hour_of_day'].apply(lambda x: 1 if 12 <= x < 17 else 0)
df['time_slot_evening'] = df['hour_of_day'].apply(lambda x: 1 if 17 <= x < 21 else 0)
df['time_slot_night'] = df['hour_of_day'].apply(lambda x: 1 if (21 <= x <= 23 or 0 <= x < 6) else 0)

# Encode 'Primary Type' and 'Time_of_Day' to integer codes
df['primary_type_encoded'] = df['Primary Type'].astype('category').cat.codes
df['time_of_day_encoded'] = df['Time_of_Day'].astype('category').cat.codes

# Map crime severity (custom as desired)
severity_map = {'BATTERY': 2, 'ASSAULT': 3, 'THEFT': 1, 'CRIMINAL': 2}
df['crime_severity_score'] = df['Primary Type'].map(severity_map).fillna(1)

# Calculate crime count and hotspot by location
loc_counts = df.groupby(['Latitude', 'Longitude'])['Primary Type'].transform('count')
df['crime_count'] = loc_counts
df['is_hotspot'] = (loc_counts > 10).astype(int)

# If missing, fill with zero
if 'crime_severity_score_block' not in df.columns:
    df['crime_severity_score_block'] = 0
if 'risk_score' not in df.columns:
    df['risk_score'] = 0

# Select ONLY numeric/encoded columns for modeling
final_features = [
    'Latitude', 'Longitude', 'primary_type_encoded', 'crime_severity_score',
    'crime_severity_score_block', 'crime_count', 'risk_score', 'is_hotspot',
    'time_of_day_encoded', 'hour_of_day', 'day_of_week',
    'time_slot_morning', 'time_slot_afternoon', 'time_slot_evening', 'time_slot_night'
]

df_model = df[final_features]

# Save new file (make sure old file is closed)
df_model.to_csv('features_for_model_final.csv', index=False)

# Optional: Show top rows to check output
print(df_model.head())
