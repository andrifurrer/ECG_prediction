## Defines basic functions use for creating the model
import os
import pandas as pd
import torch
import torchinfo
from torchinfo import summary
import dtw
from scipy.signal import butter, filtfilt
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler


def select_device():
    '''
    Check for cuda, mps or cpu and set torch default to cuda if cuda is found
    '''
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.set_default_device(device)
    # Set all tensores to GPU by default
    # if device == "cuda":
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)

    print(f"Using {device} device")
    return device


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    '''
    Defines a bandpass filter with nyq being the nyquist, lowcut the lowpass and highcut the highpass frequency. The order markes the order of the bandpass filter.
    '''
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def data_loader_filtered():
    '''
    Read in csv file of the entire dataset. If not already created, read-in all individual csv files for all subjects and actions. 
    Add subject and action column for labeling the actions. Filter the signals with an appropriate bandpass filter.
    '''
    if os.path.exists('../data/Finger/csv/finger_dataset_filtered.csv'):
        # Reading the CSV file
        df_filtered = pd.read_csv(
                    '../data/Finger/csv/finger_dataset_filtered.csv',
                    sep=',',           
                    header=0,          
                    na_values=['NA', '']  
        )
        return df_filtered
    
    else:                    
        actions = ["run", "sit", "walk"]
        subjects = [i for i in range(1 , 23)]
        df_data = pd.DataFrame()
        for subject in subjects:
            for action in actions:
                file_path = f'../data/Finger/csv/s{str(subject)}_{action}.csv'
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
        fs = 500  
        lowcut = 0.4
        highcut = 10
        df_filtered['ecg']  = bandpass_filter(df_data['ecg'], lowcut, highcut, fs, order=4)
        df_filtered['red ppg'] = bandpass_filter(df_data['pleth_4'], lowcut, highcut, fs, order=4)
        df_filtered['ir ppg'] = bandpass_filter(df_data['pleth_5'], lowcut, highcut, fs, order=4)
        df_filtered['green ppg'] = bandpass_filter(df_data['pleth_6'], lowcut, highcut, fs, order=4)
        df_filtered['action'] = df_data['action']
        df_filtered['subject'] = df_data['subject']
        # Store the DataFrame to a csv file
        df_filtered.to_csv('../data/Finger/csv/finger_dataset_filtered.csv', index=False)
        return df_filtered


def data_loader_original():
    '''
    Same as "data_loader_filtered" but without applying a bandpass filter to get the original signal values.  
    Read in csv file of the entire dataset. If not already created, read-in all individual csv files for all subjects and actions. 
    Add subject and action column for labeling the actions.
    '''
    if os.path.exists('../data/Finger/csv/finger_dataset_original.csv'):
        # Reading the CSV file
        df_original = pd.read_csv(
                    '../data/Finger/csv/finger_dataset_original.csv',
                    sep=',',           
                    header=0,          
                    na_values=['NA', '']  
        )
        return df_original
    
    else:                    
        actions = ["run", "sit", "walk"]
        subjects = [i for i in range(1 , 23)]
        df_data = pd.DataFrame()
        for subject in subjects:
            for action in actions:
                file_path = f'../data/Finger/csv/s{str(subject)}_{action}.csv'
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
        df_original['ecg']  = df_data['ecg']
        df_original['red ppg'] = df_data['pleth_4']
        df_original['ir ppg'] = df_data['pleth_5']
        df_original['green ppg'] = df_data['pleth_6']
        df_original['action'] = df_data['action']
        df_original['subject'] = df_data['subject']
        # Store the DataFrame to a csv file
        df_original.to_csv('../data/Finger/csv/finger_dataset_original.csv', index=False)
        return df_original
    

def data_loader_filtered_single(subject, action):
    '''
    Automate input reading: select subject, action
    Read in csv file and filter with a bandpass filter.
    '''
    # Reading the CSV file
    df_data = pd.read_csv(
        '../data/Finger/csv/s'+ str(subject) + '_' + str(action) + '.csv',
        sep=',',          
        header=0,          
        na_values=['NA', ''],  
    )
    # Define sampling frequency, bandpass range and apply bandpass filter
    df_filtered_single = pd.DataFrame()
    fs = 500  
    lowcut = 0.4
    highcut = 10
    df_filtered_single['ecg'] = bandpass_filter(df_data['ecg'], lowcut, highcut, fs, order=4)
    df_filtered_single['red ppg'] = bandpass_filter(df_data['pleth_4'], lowcut, highcut, fs, order=4)
    df_filtered_single['ir ppg'] = bandpass_filter(df_data['pleth_5'], lowcut, highcut, fs, order=4)
    df_filtered_single['green ppg'] = bandpass_filter(df_data['pleth_6'], lowcut, highcut, fs, order=4)
    return df_filtered_single


def data_loader_original_single(subject, action):
    '''
    Same as "data_loader_single_filtered" but without applying a bandpass filter to get the original signal values.  
    Automate input reading: select subject, action
    Read in csv file and filter with a bandpass filter.
    '''
    # Reading the CSV file
    df_data = pd.read_csv(
        '../data/Finger/csv/s'+ str(subject) + '_' + str(action) + '.csv',
        sep=',',          
        header=0,          
        na_values=['NA', ''],  
    )
    # Define sampling frequency, bandpass range and apply bandpass filter
    df_original_single = pd.DataFrame()
    df_original_single['ecg'] = df_data['ecg']
    df_original_single['red ppg'] = df_data['pleth_4']
    df_original_single['ir ppg'] = df_data['pleth_5']
    df_original_single['green ppg'] = df_data['pleth_6']
    return df_original_single

def data_loader_filtered_imu():
    '''
    Read in csv file of the entire dataset. If not already created, read-in all individual csv files for all subjects and actions. 
    Add subject and action column for labeling the actions. Filter the signals with an appropriate bandpass filter.
    '''
    if os.path.exists('../data/Finger/csv/finger_dataset_filtered_imu.csv'):
        # Reading the CSV file
        df_filtered = pd.read_csv(
                    '../data/Finger/csv/finger_dataset_filtered_imu.csv',
                    sep=',',           
                    header=0,          
                    na_values=['NA', '']  
        )
        return df_filtered
    
    else:                    
        actions = ["run", "sit", "walk"]
        subjects = [i for i in range(1 , 23)]
        df_data = pd.DataFrame()
        for subject in subjects:
            for action in actions:
                file_path = f'../data/Finger/csv/s{str(subject)}_{action}.csv'
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
        fs = 500  
        lowcut = 0.4
        highcut = 10
        df_filtered['ecg']  = bandpass_filter(df_data['ecg'], lowcut, highcut, fs, order=4)
        df_filtered['red ppg'] = bandpass_filter(df_data['pleth_4'], lowcut, highcut, fs, order=4)
        df_filtered['ir ppg'] = bandpass_filter(df_data['pleth_5'], lowcut, highcut, fs, order=4)
        df_filtered['green ppg'] = bandpass_filter(df_data['pleth_6'], lowcut, highcut, fs, order=4)
        df_filtered['action'] = df_data['action']
        df_filtered['subject'] = df_data['subject']
        df_filtered['a_x'] = df_data['a_x']
        df_filtered['a_y'] = df_data['a_y']
        df_filtered['a_z'] = df_data['a_z']
        df_filtered['g_x'] = df_data['g_x']
        df_filtered['g_y'] = df_data['g_y']
        df_filtered['g_z'] = df_data['g_z']
        # Store the DataFrame to a csv file
        df_filtered.to_csv('../data/Finger/csv/finger_dataset_filtered_imu.csv', index=False)
        return df_filtered
    

def normalization_group_action(df):
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
        ppg_normalized = scaler_input.fit_transform(group[['red ppg', 'ir ppg', 'green ppg']])

        # Fit and transform the ECG column (target)
        ecg_normalized = scaler_target.fit_transform(group[['ecg']])

        # Save the scalers for this subject-action group
        scalers[(subject, action)] = {'input_scaler': scaler_input, 'target_scaler': scaler_target}
        # Inspect original scalers (collective normalization)
        print(scalers[(subject, action)]['input_scaler'].data_min_, scalers[(subject, action)]['input_scaler'].data_max_)

        # Create a copy of the group with normalized values
        group_normalized = group.copy()
        group_normalized[['red ppg', 'ir ppg', 'green ppg']] = ppg_normalized
        group_normalized[['ecg']] = ecg_normalized

        # Append the normalized group to the list
        normalized_data.append(group_normalized)

    # Combine all normalized groups back into a single DataFrame
    normalized_df = pd.concat(normalized_data).reset_index(drop=True)
    return normalized_df, scalers


def global_normalization(df):
    '''
    Apply global normalization to the entire dataset across all subjects and actions.
    '''
    # Initialize scalers for global normalization
    scaler_input = MinMaxScaler(feature_range=(-1, 1))  # For PPG signals
    scaler_target = MinMaxScaler(feature_range=(-1, 1))  # For ECG signals

    # Normalize the PPG columns (inputs)
    ppg_columns = ['red ppg', 'ir ppg', 'green ppg']
    df[ppg_columns] = scaler_input.fit_transform(df[ppg_columns])

    # Normalize the ECG column (target)
    df['ecg'] = scaler_target.fit_transform(df[['ecg']])

    # Store the scalers for inspection or future transformations
    scalers = {
        'input_scaler': scaler_input,
        'target_scaler': scaler_target
    }

    return df, scalers

def normalization_group_action_imu(df):
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

        # Fit and transform the PPG and IMU columns (inputs)
        ppg_normalized = scaler_input.fit_transform(group[['red ppg', 'ir ppg', 'green ppg', 'a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']])

        # Fit and transform the ECG column (target)
        ecg_normalized = scaler_target.fit_transform(group[['ecg']])

        # Save the scalers for this subject-action group
        scalers[(subject, action)] = {'input_scaler': scaler_input, 'target_scaler': scaler_target}
        # Inspect original scalers (collective normalization)
        print(scalers[(subject, action)]['input_scaler'].data_min_, scalers[(subject, action)]['input_scaler'].data_max_)

        # Create a copy of the group with normalized values
        group_normalized = group.copy()
        group_normalized[['red ppg', 'ir ppg', 'green ppg', 'a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']] = ppg_normalized
        group_normalized[['ecg']] = ecg_normalized

        # Append the normalized group to the list
        normalized_data.append(group_normalized)

    # Combine all normalized groups back into a single DataFrame
    normalized_df = pd.concat(normalized_data).reset_index(drop=True)
    return normalized_df, scalers

def sequences(df, sequence_length, sequence_step_size, subset):
    '''
    Create sequnces with the per (subject, action) pair normalized dataframe
    '''
    # Retrieve input ppg signals
    input_columns = ['red ppg', 'ir ppg', 'green ppg']
    x_normalized = df[input_columns].values

    # Retrieve target ecg signals
    y_normalized = df[['ecg']].values

    # Convert to PyTorch tensors
    x_data = torch.tensor(x_normalized, dtype=torch.float32)  # Shape: [samples, 3]
    y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1]

    # Reshape for sequence length and adjustable stepsize. Sequences are shifted by timestamp / sample stepsize per sequence! 
    num_sequences = len(df) - sequence_length + 1

    x_sequences = torch.stack([x_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 3]
    y_sequences = torch.stack([y_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 1]
    return x_sequences, y_sequences


def sequences_imu(df, sequence_length, sequence_step_size, subset):
    '''
    Create sequnces with the per (subject, action) pair normalized dataframe
    '''
    # Retrieve input ppg signals
    input_columns = ['red ppg', 'ir ppg', 'green ppg', 'a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']
    x_normalized = df[input_columns].values

    # Retrieve target ecg signals
    y_normalized = df[['ecg']].values

    # Convert to PyTorch tensors
    x_data = torch.tensor(x_normalized, dtype=torch.float32)  # Shape: [samples, 6]
    y_data = torch.tensor(y_normalized, dtype=torch.float32)  # Shape: [samples, 1]

    # Reshape for sequence length and adjustable stepsize. Sequences are shifted by timestamp / sample stepsize per sequence! 
    num_sequences = len(df) - sequence_length + 1

    x_sequences = torch.stack([x_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 6]
    y_sequences = torch.stack([y_data[i:i + sequence_length] for i in range(0, int(num_sequences*subset), int(sequence_step_size))])  # [num_sequences, seq_length, 1]
    return x_sequences, y_sequences


def downsample_sequence(sequence, factor):
    '''
    Downsamples a given input sequnce by the stated factor
    '''
    return sequence[::factor]


def compute_batched_dtw(pred, act, batch_size, downsampling_factor):
    '''
    Iterate over all sequences of the prediction and actual sequences, downsample by a given factor and calculate the DTW
    '''
    dtw_distances = []
    for start in range(0, len(pred), batch_size):
        end = start + batch_size
        pred_batch = pred[start:end]
        act_batch = act[start:end]

        # Downsample each batch
        pred_batch_downsampled = downsample_sequence(pred_batch, downsampling_factor)
        act_batch_downsampled = downsample_sequence(act_batch, downsampling_factor)

        # Compute DTW for the batch
        alignment = dtw.dtw(pred_batch_downsampled, act_batch_downsampled, keep_internals=False)
        dtw_distances.append(alignment.distance)

    # Return average DTW across batches
    return sum(dtw_distances) / len(dtw_distances)


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    '''
    Save the model and optimizer state as well as the epoch with the validation loss
    '''
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }
    torch.save(state, filepath)


def generate_square_subsequent_mask(size):
    '''
    Generate a mask, upper triangular matrix gets filled with -inf.
    '''
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))


def save_model_summary(model, X_train, y_train, device, model_name, batch_size, model_summary_folder):
    '''
    Save the model as a ONNX file and the model summary in a txt file
    '''
    X_train_sample = X_train[:1]
    y_train_sample = y_train[:1]

    # Call the torchinfo summary method
    summary_txt = summary(model, input_data=X_train_sample, depth=5, device=device)

     # Write summary to a file
    summary_file_path = f"{model_summary_folder}/{model_name}_model_summary.txt"
    with open(summary_file_path, "w") as f:
        f.write(str(summary_txt))

    print(f"Summary written to {summary_file_path}")

    ### ONNX export
    # Trace the model to optimize for ONNX export
    # sample_input = (X_train_sample.to(device), y_train_sample.to(device))
    # traced_model = torch.jit.trace(model, sample_input)

    # Export model to ONNX
    # try:
    #     torch.onnx.export(
    #         traced_model, # Traced Model to export
    #         sample_input,
    #         f"../models/model_summary/{model_name}.onnx", # Folder and filename of the ONNX file
    #         opset_version=14,
    #         verbose=False,
    #         input_names=["Input PPG signals", "Target ECG signal"], # Rename inputs for the ONNX model
    #         output_names=["Predicted ECG signal"], # Rename output for the ONNX model
    #     )
    #     print(f"ONNX model exported to {summary_file_dir}{model_name}.onnx")
    # except Exception as e:
    #     print(f"Error exporting ONNX model: {e}")

    # torch.onnx.export(
    #     model, # Model to export
    #     (X_train_sample.to(device), y_train_sample.to(device)),  # Pass both inputs as a tuple
    #     f"../models/model_summary/{model_name}.onnx", # Folder and filename of the ONNX file
    #     opset_version=14,
    #     verbose=False,
    #     input_names=["Input PPG signals", "Target ECG signal"],  # Rename inputs for the ONNX model
    #     output_names=["Predicted ECG signal"],  # Rename output for the ONNX model
    # )

# Ensure loading data on the correct device
def load_data_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)  # Move tensor to the specified device (CPU or CUDA)
    elif isinstance(data, dict):
        return {key: load_data_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [load_data_to_device(item, device) for item in data]
    else:
        return data  # For non-tensor data, return as-is