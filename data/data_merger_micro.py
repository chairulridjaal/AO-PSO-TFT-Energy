import pandas as pd
import numpy as np
import os
import sys
import holidays

os.chdir(sys.path[0])

def main():
    base_dir = r"c:\Users\Chairulridjal\Downloads\Praktikum\6 - SEM\Thesis\data"
    grid_dir = os.path.join(base_dir, "grid_raw")
    
    # 1. Load ISO-NE Excel Data
    years = [2021, 2022, 2023, 2024]
    iso_dfs = []
    
    print("Loading ISO-NE load data...")
    for year in years:
        file_path = os.path.join(grid_dir, f"{year}_smd_hourly.xlsx")
        df = pd.read_excel(file_path, sheet_name="ISO NE CA")
        iso_dfs.append(df)
        
    iso_df = pd.concat(iso_dfs, ignore_index=True)
    
    print(f"ISO-NE data loaded. Shape: {iso_df.shape}")
    
    # Keep only target columns
    cols_to_keep = ['Date', 'Hr_End', 'System_Load', 'DA_Demand', 'Dry_Bulb', 'Dew_Point']
    iso_df = iso_df[cols_to_keep].copy()
    
    # Clean possible textual rows (e.g. 'total') by coercing to numeric
    for col in ['Hr_End', 'System_Load', 'DA_Demand', 'Dry_Bulb', 'Dew_Point']:
        iso_df[col] = pd.to_numeric(iso_df[col], errors='coerce')
    
    # Drop rows that don't have a valid Hr_End or Date
    iso_df.dropna(subset=['Date', 'Hr_End', 'Dry_Bulb'], inplace=True)
    iso_df['Hr_End'] = iso_df['Hr_End'].astype(int)
    
    # 2. Unit Conversion: Fahrenheit to Celsius
    iso_df['Dry_Bulb'] = (iso_df['Dry_Bulb'] - 32) * 5/9
    iso_df['Dew_Point'] = (iso_df['Dew_Point'] - 32) * 5/9
    
    # 3. Timestamp Alignment
    print("Aligning timestamps...")
    iso_df['Date'] = pd.to_datetime(iso_df['Date'])
    
    # ISO-NE "Hour Ending" (1-24). Start of hour is Hr_End - 1 (0-23)
    start_hour = iso_df['Hr_End'] - 1
    
    date_str = iso_df['Date'].dt.strftime('%Y-%m-%d')
    time_str = start_hour.astype(str).str.zfill(2) + ':00:00'
    
    # Create the naive timestamp
    iso_df['time'] = pd.to_datetime(date_str + ' ' + time_str)
    iso_df.set_index('time', inplace=True)
    
    # 4. Load & Align Weather Data
    weather_file = os.path.join(grid_dir, "open-meteo-42.36N71.13W19m.csv")
    print(f"Loading weather data from {weather_file}...")
    
    # Skip first 10 rows (header metadata and blank line) as observed in the file
    df_weather = pd.read_csv(weather_file, skiprows=10)
    df_weather['time'] = pd.to_datetime(df_weather['time'])
    
    # Identify the exact column names Open-Meteo generated
    # (Adjust these string names if Open-Meteo named them slightly differently in your CSV)
    weather_vars = {
        'temperature_2m (°C)': 'weighted_temperature',
        'relative_humidity_2m (%)': 'weighted_humidity',
        'apparent_temperature (°C)': 'weighted_apparent_temp',
        'precipitation (mm)': 'weighted_precipitation',
        'cloud_cover (%)': 'weighted_cloud_cover',
        'shortwave_radiation_instant (W/m²)': 'weighted_solar'
    }

    # 5. Weighted Weather Aggregation for ALL variables
    print("Applying ISO-NE seasonal weights to all exogenous variables...")
    w_summ = {0: 0.201, 1: 0.070, 2: 0.046, 3: 0.058, 4: 0.085, 5: 0.049, 6: 0.277, 7: 0.214}
    w_wint = {0: 0.214, 1: 0.075, 2: 0.040, 3: 0.055, 4: 0.082, 5: 0.048, 6: 0.277, 7: 0.209}
    
    summ_arr = np.array([w_summ[i] for i in range(8)])
    wint_arr = np.array([w_wint[i] for i in range(8)])
    
    # We will build a list of merged dataframes to join later
    final_df = iso_df.copy()
    
    for raw_col, new_col_name in weather_vars.items():
        if raw_col in df_weather.columns:
            # Pivot just this specific weather variable
            pivot = df_weather.pivot(index='time', columns='location_id', values=raw_col)
            
            # Vectorized application
            is_summer = pivot.index.month.isin([6, 7, 8, 9])
            val_summ = pivot.dot(summ_arr)
            val_wint = pivot.dot(wint_arr)
            
            # Create the weighted column
            weighted_series = np.where(is_summer, val_summ, val_wint)
            pivot[new_col_name] = weighted_series
            
            # Join it to our final dataframe
            final_df = final_df.join(pivot[[new_col_name]], how='inner')
        else:
            print(f"WARNING: {raw_col} not found in Open-Meteo CSV.")
    
    # 6. Feature Engineering
    print("Adding features...")
    hour = final_df.index.hour
    month = final_df.index.month
    final_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    final_df['month_sin'] = np.sin(2 * np.pi * month / 12)
    final_df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    us_holidays = holidays.US(years=years)
    final_df['is_holiday'] = final_df.index.strftime('%Y-%m-%d').map(lambda x: 1 if x in us_holidays else 0)
    final_df['day_of_week'] = final_df.index.dayofweek # 0-6 (Monday=0)
    
    final_df['temp_check'] = final_df['weighted_temperature'] - final_df['Dry_Bulb']
    
    # 7. Output Requirements
    # Drop unneeded columns and rename index to match micro_household_dataset
    if 'Date' in final_df.columns:
        final_df.drop(columns=['Date'], inplace=True)
    if 'Hr_End' in final_df.columns:
        final_df.drop(columns=['Hr_End'], inplace=True)
        
    final_df.index.name = 'Datetime'
    
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    final_rows = len(final_df)
    
    out_path = os.path.join(base_dir, "macro_grid_dataset.csv")
    final_df.to_csv(out_path)
    
    print("\n" + "="*50)
    print(" " * 15 + "SUMMARY REPORT")
    print("="*50)
    print(f"Total row count out of merge  : {initial_rows}")
    print(f"Final row count (after dropna): {final_rows}")
    print(f"Date Range Coverage           : {final_df.index.min()} to {final_df.index.max()}")
    
    if final_rows > 0:
        mean_diff = final_df['temp_check'].abs().mean()
        max_diff = final_df['temp_check'].abs().max()
        print(f"\nUnit Conversion & Validation Check:")
        print(f"-> Average absolute diff (Weighted Meteo vs ISO-NE): {mean_diff:.2f} °C")
        print(f"-> Max absolute diff: {max_diff:.2f} °C")
        if mean_diff < 5.0:
            print("-> SUCCESS: Temperature alignment looks accurate!")
        else:
            print("-> WARNING: Large temperature difference. Verify alignment.")
    
    print(f"\nDataset exported to:")
    print(f" {out_path}")
    print("="*50)

if __name__ == '__main__':
    main()