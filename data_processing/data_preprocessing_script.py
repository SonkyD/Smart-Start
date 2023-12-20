# Import Libraries
import pandas as pd 
import numpy as np
import datetime as dt
from datetime import datetime, timezone, timedelta
import random 
import warnings
warnings.filterwarnings("ignore")

# Read the Data and safe as DataFrame
df_raw_train_series = pd.read_parquet('../data/train_series.parquet')
df_raw_train_event = pd.read_csv('../data/train_events.csv')

# Convert timestamp to datetime
df_raw_train_series['timestamp'] = pd.to_datetime(df_raw_train_series['timestamp'])
df_raw_train_event['timestamp'] = pd.to_datetime(df_raw_train_event['timestamp'])

# drop NaN values of event data
df_raw_train_event.dropna(axis= 0, inplace=True)

#Calculating the slope of the linear regression
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum([xi*yi for xi, yi in zip(x, y)])
    sum_x_squared = sum([xi**2 for xi in x])

    slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x_squared - sum_x**2)
    return slope

# Get all unique IDs from 'series_id' column of raw data
unique_ids = df_raw_train_series['series_id'].unique()

# Define time window for random start and end time in minutes
mini = 60
max = 180

# Processing starts here for each unique ID
for id in unique_ids:
    # Processing starts here for each unique ID, convertion to datetime and setting timestamp as index
    id_raw_train = df_raw_train_series[df_raw_train_series['series_id'] == id]
    id_raw_event = df_raw_train_event[df_raw_train_event['series_id'] == id]

    id_raw_train['timestamp'] = pd.to_datetime(id_raw_train['timestamp']) 
    id_raw_event['timestamp'] = pd.to_datetime(id_raw_event['timestamp']) 

    id_raw_train.set_index('timestamp', inplace=True)

   # Lists for numerical and non-numerical data
    numerical = ['anglez', 'enmo', 'step']
    non_numerical = ['series_id']

## Feature engineering 

    # Resampling the data to 1-minute intervals and calculating mean and standard deviation for numerical data
    binned_numeric_data_mean = id_raw_train[numerical].resample('1T').mean()
    numerical.remove('step') # Remove 'step' from numerical list before calculating standard deviation
    binned_numeric_data_std = id_raw_train[numerical].resample('1T').std() 
    binned_numeric_data_std = binned_numeric_data_std.add_suffix('_std') # Adding suffix for standard deviation columns
    binned_numeric_data = pd.concat([binned_numeric_data_mean, binned_numeric_data_std], axis=1) 
    binned_non_numeric_data = id_raw_train[non_numerical].resample('1T').first() # Handling non-numerical data
    binned_id = pd.concat([binned_numeric_data, binned_non_numeric_data], axis=1) 
 
 
    # Slope calculation for ENMO and Anglez

    # Initialize lists to store slope, counters and differences
    slope_enmo = []
    slope_anglez = []
    anglez_outside_45_counter = []
    enmo_over_008_counter = []
    anglez_difference_5_per_min = []

    # Loop over the data in 1 minute intervals (12 rows per minute) and calculate slope and counter for ENMO and Anglez
    for i in range(0, len(id_raw_train), 12):
        # Calculate Slope for ENMO and Anglez
        slope_enmo.append(linear_regression(id_raw_train['step'][i:i+12], id_raw_train['enmo'][i:i+12]))
        slope_anglez.append(linear_regression(id_raw_train['step'][i:i+12], id_raw_train['anglez'][i:i+12]))
        
        #Check if Anglez difference is greater than 5 degree
        anglez_dif_5_counter = 0
        for k in range(i, i+12):
            if (abs(id_raw_train['anglez'][k] - id_raw_train['anglez'][k-1])) > 5:
                anglez_dif_5_counter += 1
        anglez_difference_5_per_min.append(anglez_dif_5_counter)
        
        # Adding counter for ENMO and Anglez to calculate how often it goes over threshold (Enmo = 0.008), (Anglez = -45, +45)
        # Threshold was chosen based on EDA Data
        enmo_counter = 0
        anglez_counter = 0
        
        # Counts instances where enmo over 0.008
        for j in range(i, i+12):
            if id_raw_train['enmo'][j] > 0.008:
                enmo_counter += 1
        enmo_over_008_counter.append(enmo_counter)
        
        # Counts instances where AngleZ is outside the range -45 to 45
        for j in range(i, i+12):
            if id_raw_train['anglez'][j] < -45 or id_raw_train['anglez'][j] > 45:
                anglez_counter += 1 
        anglez_outside_45_counter.append(anglez_counter)
       
    # Add more calculated features to the dataframe     
    binned_id['slope_enmo'] = slope_enmo
    binned_id['slope_anglez'] = slope_anglez
    binned_id['anglez_outside_45_counter'] = anglez_outside_45_counter
    binned_id['enmo_over_008_counter'] = enmo_over_008_counter
    binned_id['anglez_difference_5_per_min'] = anglez_difference_5_per_min
    
    # Windows for rolling calculations for each variable and adding the mean and std for each window
    windows = {'5min': 5, '10min': 10}
    for window_name, window_size in windows.items():
        for variable in ['anglez', 'enmo']:  
            # Rolling calculations for each variable
            rolling_window = binned_id[variable].rolling(window=window_size)
            binned_id[f'{variable}_{window_name}_mean'] = rolling_window.mean()
            binned_id[f'{variable}_{window_name}_std'] = rolling_window.std()

    # Shift the mean and std by 5min and 10min respectively
            binned_id[f'{variable}_{window_name}_mean_shifted'] = binned_id[f'{variable}_{window_name}_mean'].shift(periods=window_size)
            binned_id[f'{variable}_{window_name}_std_shifted'] = binned_id[f'{variable}_{window_name}_std'].shift(periods=window_size)
    
    # Calculating the ratio of anglez and enmo and replacing divisions with NaN
    binned_id['anglez_enmo_ratio'] = binned_id['anglez'] / binned_id['enmo'].replace(0, np.nan)
    
    # Preparing event data
    id_raw_event.drop(columns=['step', 'series_id'], inplace=True)
    id_raw_event.reset_index(drop=True, inplace=True)
    merge_id = pd.merge(binned_id, id_raw_event, on='timestamp', how='left')
    merge_id['timestamp'] = pd.to_datetime(merge_id['timestamp']) 
    merge_id.set_index('timestamp', inplace=True)
    
    
    # Iterating through each unique night in merged data
    for night in merge_id['night'].dropna().unique():
        #Determine onset and wakeup time for each night
        onset_time = merge_id[(merge_id['night'] == night) & (merge_id['event'] == 'onset')].index.min()
        wakeup_time = merge_id[(merge_id['night'] == night) & (merge_id['event'] == 'wakeup')].index.max()
        #Generate a random start and end time for each night in the defined time frame (mini, max)
        if pd.notnull(onset_time) and pd.notnull(wakeup_time):
            random_start = random.randint(mini, max)
            random_end = random.randint(mini, max)
            #Define start/end for data slice
            start_time = onset_time - pd.Timedelta(minutes=random_start)
            end_time = wakeup_time + pd.Timedelta(minutes=random_end)

            # Filtering data within the time window
            filtered_data = merge_id.loc[start_time:end_time]
            filtered_data['event'] = filtered_data['event'].fillna('awake') # Set default event to 'awake'

            # Check if 'onset' and 'wakeup' events exist in filtered data
            if 'onset' in filtered_data['event'].values and 'wakeup' in filtered_data['event'].values:
                row_number_of_onset = filtered_data.index.get_loc(filtered_data[filtered_data['event'] == 'onset'].index[0])
                row_number_of_wakeup = filtered_data.index.get_loc(filtered_data[filtered_data['event'] == 'wakeup'].index[0])
                # Label all rows between 'onset' and 'wakeup' as 'sleep'
                for i in range(row_number_of_onset + 1, row_number_of_wakeup):
                    filtered_data.iloc[i, filtered_data.columns.get_loc('event')] = 'sleep'
                    
                    
            # Adding minutes_since_onset calculation
            minutes_since_onset = 0
            onset_flag = False
            
            # Iterating through each row in filtered data
            for index, row in filtered_data.iterrows():
                if row['event'] == 'onset':
                    onset_flag = True
                    minutes_since_onset = 0  # Reset counter at onset
                elif row['event'] == 'sleep' or row['event'] == 'awake' and onset_flag == True:
                    minutes_since_onset += 1  # Increment counter after onset

                filtered_data.loc[index, 'minutes_since_onset'] = minutes_since_onset
                
                
            # Extract year, month, day, hour, and minute from the timestamp
                filtered_data.loc[index, 'year'] = index.year
                filtered_data.loc[index, 'month'] = index.month
                filtered_data.loc[index, 'day'] = index.day
                filtered_data.loc[index, 'hour'] = index.hour
                filtered_data.loc[index, 'minute'] = index.minute           
                    
            # Exporting the data to Parquet files
            filename = f'../data/file_per_night/patient_{id}_{night}_random_{mini}_{max}_minutes.parquet'
            filtered_data.to_parquet(filename)
            print(f' Exported {filename} successfully')

print('Processing completed for all IDs')
