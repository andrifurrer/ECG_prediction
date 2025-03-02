import os
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import torch 


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    '''
    Defines a bandpass filter with nyq being the nyquist, lowcut the lowpass and highcut the highpass frequency. The order markes the order of the bandpass filter.
    '''
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def data_loader_filtered_wrist():
    '''
    Read in csv file of the entire dataset. If not already created, read-in all individual csv files for all subjects and actions. 
    Add subject and action column for labeling the actions. Filter the ecg signals with an appropriate bandpass filter, the pgg signals are already prefiltered.
    '''
    if os.path.exists('../data/Wrist_scv/wrist_dataset_filtered.csv'):
        # Reading the CSV file
        df_filtered = pd.read_csv(
                    '../data/Wrist_csv/wrist_dataset_filtered.csv',
                    sep=',',           
                    header=0,          
                    na_values=['NA', '']  
        )
        return df_filtered
    
    else:                    
        actions = ["run", "walk", "low_resistance_bike", "high_resistance_bike"]
        subjects = [i for i in range(1 , 10)]
        df_data = pd.DataFrame()
        for subject in subjects:
            for action in actions:
                file_path = f'../data/Wrist_csv/s{str(subject)}_{action}.csv'
                # Reading the CSV file
                try:
                    df_temp = pd.read_csv(
                        file_path,
                        sep=',',           
                        header=0,          
                        na_values=['NA', '']  
                    )
                    # Adding a column with the current action and subject
                    df_temp['action'] = action
                    df_temp['subject'] = subject
                    df_data = pd.concat([df_data, df_temp], ignore_index=True)

                except FileNotFoundError:
                    print(f"File not found: {file_path}")
        # Define sampling frequency, bandpass range and apply bandpass filter
        df_filtered = pd.DataFrame()
        fs = 256  
        lowcut = 0.4
        highcut = 10
        df_filtered['ecg'] = bandpass_filter(df_data['chest_ecg'], lowcut, highcut, fs, order=4)
        df_filtered['ppg'] = df_data['wrist_ppg']
        df_filtered['action'] = df_data['action']
        df_filtered['subject'] = df_data['subject']
        # Store the DataFrame to a csv file
        df_filtered.to_csv('../data/Wrist_csv/wrist_dataset_filtered.csv', index=False)
        # Drop rows containing NaN values
        df_filtered = df_filtered.dropna()
        return df_filtered
    
def data_loader_filtered_wrist_imu():
    '''
    Read in csv file of the entire dataset. If not already created, read-in all individual csv files for all subjects and actions. 
    Add subject and action column for labeling the actions. Filter the ecg signals with an appropriate bandpass filter, the pgg signals are already prefiltered.
    '''
    if os.path.exists('../data/Wrist_scv/wrist_dataset_filtered_imu.csv'):
        # Reading the CSV file
        df_filtered = pd.read_csv(
                    '../data/Wrist_csv/wrist_dataset_filtered_imu.csv',
                    sep=',',           
                    header=0,          
                    na_values=['NA', '']  
        )
        return df_filtered
    
    else:                    
        actions = ["run", "walk", "low_resistance_bike", "high_resistance_bike"]
        subjects = [i for i in range(1 , 10)]
        df_data = pd.DataFrame()
        for subject in subjects:
            for action in actions:
                file_path = f'../data/Wrist_csv/s{str(subject)}_{action}.csv'
                # Reading the CSV file
                try:
                    df_temp = pd.read_csv(
                        file_path,
                        sep=',',           
                        header=0,          
                        na_values=['NA', '']  
                    )
                    # Adding a column with the current action and subject
                    df_temp['action'] = action
                    df_temp['subject'] = subject
                    df_data = pd.concat([df_data, df_temp], ignore_index=True)

                except FileNotFoundError:
                    print(f"File not found: {file_path}")
        # Define sampling frequency, bandpass range and apply bandpass filter
        df_filtered = pd.DataFrame()
        fs = 256  
        lowcut = 0.4
        highcut = 10
        df_filtered['ecg'] = bandpass_filter(df_data['chest_ecg'], lowcut, highcut, fs, order=4)
        df_filtered['ppg'] = df_data['wrist_ppg']
        df_filtered['action'] = df_data['action']
        df_filtered['subject'] = df_data['subject']
        df_filtered['a_x_low'] = df_data['wrist_low_noise_accelerometer_x']
        df_filtered['a_y_low'] = df_data['wrist_low_noise_accelerometer_y']
        df_filtered['a_z_low'] = df_data['wrist_low_noise_accelerometer_z']
        df_filtered['a_x_wide'] = df_data['wrist_wide_range_accelerometer_x']
        df_filtered['a_y_wide'] = df_data['wrist_wide_range_accelerometer_y']
        df_filtered['a_z_wide'] = df_data['wrist_wide_range_accelerometer_z']
        df_filtered['g_x'] = df_data['wrist_gyro_x']
        df_filtered['g_y'] = df_data['wrist_gyro_y']
        df_filtered['g_z'] = df_data['wrist_gyro_z']
        # Store the DataFrame to a csv file
        df_filtered.to_csv('../data/Wrist_csv/wrist_dataset_filtered_imu.csv', index=False)
        # Drop rows containing NaN values
        df_filtered = df_filtered.dropna()
        return df_filtered


def data_loader_original_wrist():
    '''
    Read in csv file of the entire dataset. If not already created, read-in all individual csv files for all subjects and actions. 
    Add subject and action column for labeling the actions.
    '''
    if os.path.exists('../data/Wrist_scv/wrist_dataset_originial.csv'):
        # Reading the CSV file
        df_original = pd.read_csv(
                    '../data/Wrist_csv/wrist_dataset_originial.csv',
                    sep=',',           
                    header=0,          
                    na_values=['NA', '']  
        )
        return df_original
    
    else:                    
        actions = ["run", "walk", "low_resistance_bike", "high_resistance_bike"]
        subjects = [i for i in range(1 , 10)]
        df_data = pd.DataFrame()
        for subject in subjects:
            for action in actions:
                file_path = f'../data/Wrist_csv/s{str(subject)}_{action}.csv'
                # Reading the CSV file
                try:
                    df_temp = pd.read_csv(
                        file_path,
                        sep=',',           
                        header=0,          
                        na_values=['NA', '']  
                    )
                    # Adding a column with the current action and subject
                    df_temp['action'] = action
                    df_temp['subject'] = subject
                    df_data = pd.concat([df_data, df_temp], ignore_index=True)

                except FileNotFoundError:
                    print(f"File not found: {file_path}")
        # Define sampling frequency, bandpass range and apply bandpass filter
        df_original = pd.DataFrame()
        df_original['ecg']  = df_data['chest_ecg']
        df_original['ppg'] = df_data['wrist_ppg']
        df_original['action'] = df_data['action']
        df_original['subject'] = df_data['subject']
        # Store the DataFrame to a csv file
        df_original.to_csv('../data/Wrist_csv/wrist_dataset_originial.csv', index=False)
        # Drop rows containing NaN values
        df_original = df_original.dropna()
        return df_original
    

def global_normalization_wrist(original_df):
    '''
    Apply global normalization to the entire dataset across all subjects and actions.
    '''
    # Initialize scalers for global normalization
    scaler_input = MinMaxScaler(feature_range=(-1, 1))  # For PPG signals
    scaler_target = MinMaxScaler(feature_range=(-1, 1))  # For ECG signals

    # Work with a copy to avoid modifying the original DataFrame
    normalized_df = original_df.copy()

    # Normalize the PPG columns (inputs)
    ppg_columns = ['ppg']
    normalized_df[ppg_columns] = scaler_input.fit_transform(normalized_df[ppg_columns])

    # Normalize the ECG column (target)
    normalized_df['ecg'] = scaler_target.fit_transform(normalized_df[['ecg']])

    # Store the scalers for inspection or future transformations
    scalers = {
        'input_scaler': scaler_input,
        'target_scaler': scaler_target
    }

    return normalized_df, scalers

def normalization_group_action_wrist(df):
    '''
    Group the data by subject and actions, and normalize each (subject, action) pair
    '''
    # Initialize a dictionary to store scalers for each subject-action group
    scalers = {}

    # Placeholder for normalized data
    normalized_data = []

    # Group data by subject and action
    grouped_data = df.groupby(['subject', 'action'])
    print(grouped_data.first())

    for (subject, action), group in grouped_data:
        # Initialize scalers for PPG (inputs) and ECG (targets)
        scaler_input = MinMaxScaler(feature_range=(-1, 1))  # For PPG signals
        scaler_target = MinMaxScaler(feature_range=(-1, 1))  # For ECG signals

        # Fit and transform the PPG columns (inputs)
        ppg_normalized = scaler_input.fit_transform(group[['ppg']])

        # Fit and transform the ECG column (target)
        ecg_normalized = scaler_target.fit_transform(group[['ecg']])

        # Save the scalers for this subject-action group
        scalers[(subject, action)] = {'input_scaler': scaler_input, 'target_scaler': scaler_target}
        # Inspect original scalers (collective normalization)
        print(scalers[(subject, action)]['input_scaler'].data_min_, scalers[(subject, action)]['input_scaler'].data_max_)

        # Create a copy of the group with normalized values
        group_normalized = group.copy()
        group_normalized[['ppg']] = ppg_normalized
        group_normalized[['ecg']] = ecg_normalized

        # Append the normalized group to the list
        normalized_data.append(group_normalized)

    # Combine all normalized groups back into a single DataFrame
    normalized_df = pd.concat(normalized_data).reset_index(drop=True)
    return normalized_df, scalers

def normalization_group_action_wrist_imu(df):
    '''
    Group the data by subject and actions, and normalize each (subject, action) pair
    '''
    # Initialize a dictionary to store scalers for each subject-action group
    scalers = {}

    # Placeholder for normalized data
    normalized_data = []

    # Group data by subject and action
    grouped_data = df.groupby(['subject', 'action'])
    print(grouped_data.first())

    for (subject, action), group in grouped_data:
        # Initialize scalers for PPG (inputs) and ECG (targets)
        scaler_input = MinMaxScaler(feature_range=(-1, 1))  # For PPG and IMU signals
        scaler_target = MinMaxScaler(feature_range=(-1, 1))  # For ECG signals

        # Fit and transform the PPG columns (inputs)
        ppg_normalized = scaler_input.fit_transform(group[['ppg', 'a_x_low', 'a_y_low', 'a_z_low', 'a_x_wide', 'a_y_wide', 'a_z_wide', 'g_x', 'g_y', 'g_z']])

        # Fit and transform the ECG column (target)
        ecg_normalized = scaler_target.fit_transform(group[['ecg']])

        # Save the scalers for this subject-action group
        scalers[(subject, action)] = {'input_scaler': scaler_input, 'target_scaler': scaler_target}
        # Inspect original scalers (collective normalization)
        print(scalers[(subject, action)]['input_scaler'].data_min_, scalers[(subject, action)]['input_scaler'].data_max_)

        # Create a copy of the group with normalized values
        group_normalized = group.copy()
        group_normalized[['ppg', 'a_x_low', 'a_y_low', 'a_z_low', 'a_x_wide', 'a_y_wide', 'a_z_wide', 'g_x', 'g_y', 'g_z']] = ppg_normalized
        group_normalized[['ecg']] = ecg_normalized

        # Append the normalized group to the list
        normalized_data.append(group_normalized)

    # Combine all normalized groups back into a single DataFrame
    normalized_df = pd.concat(normalized_data).reset_index(drop=True)
    return normalized_df, scalers


def sequences_wrist(df, sequence_length, sequence_step_size, subset):
    '''
    Create sequnces with the per (subject, action) pair normalized dataframe
    '''
    # Retrieve input ppg signals
    input_columns = ['ppg']
    x_normalized = df[input_columns].values

    # Retrieve target ecg signals
    y_normalized = df[['ecg']].values

    # Convert to PyTorch tensors
    x_data = torch.tensor(x_normalized, dtype=torch.float32)  # Shape: [samples, 1]
    y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1]

    # Reshape for sequence length and adjustable stepsize. Sequences are shifted by timestamp / sample stepsize per sequence! 
    num_sequences = len(df) - sequence_length + 1

    x_sequences = torch.stack([x_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 3]
    y_sequences = torch.stack([y_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 1]
    return x_sequences, y_sequences


def sequences_wrist_seq2point(df, sequence_length):
    '''
    Create sequnces with the per (subject, action) pair normalized dataframe, where each input ppg sequence has a corresponding ecg target
    '''
    # Retrieve input ppg signals
    x_normalized = df['ppg'].values

    # Retrieve target ecg signals
    y_normalized = df[['ecg']].values

    # Convert to PyTorch tensors
    x_data = torch.tensor(x_normalized, dtype=torch.float32)  # Shape: [samples, 1]
    y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1]

    # Reshape for sequence length and adjustable stepsize. Sequences are shifted by timestamp / sample stepsize per sequence! 
    num_sequences = len(df) - sequence_length + 1

    x_sequences = torch.stack([x_data[i:i + sequence_length] for i in range(len(x_data)-sequence_length)]).unsqueeze(-1) # [num_sequences, seq_length, 3]
    y_sequences = torch.stack([y_data[i + sequence_length] for i in range(len(y_data)-sequence_length)]).unsqueeze(-1)  # [num_sequences, seq_length, 1]
    return x_sequences, y_sequences

def sequences_wrist_imu(df, sequence_length, sequence_step_size, subset):
    '''
    Create sequnces with the per (subject, action) pair normalized dataframe
    '''
    # Retrieve input ppg signals
    input_columns = ['ppg', 'a_x_low', 'a_y_low', 'a_z_low', 'a_x_wide', 'a_y_wide', 'a_z_wide', 'g_x', 'g_y', 'g_z']
    x_normalized = df[input_columns].values

    # Retrieve target ecg signals
    y_normalized = df[['ecg']].values

    # Convert to PyTorch tensors
    x_data = torch.tensor(x_normalized, dtype=torch.float32)  # Shape: [samples, 7]
    y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1]

    # Reshape for sequence length and adjustable stepsize. Sequences are shifted by timestamp / sample stepsize per sequence! 
    num_sequences = len(df) - sequence_length + 1

    x_sequences = torch.stack([x_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 7]
    y_sequences = torch.stack([y_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 1]
    return x_sequences, y_sequences