import os
import io
import sys
import psutil
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchinfo
from torchinfo import summary
from torch.utils.checkpoint import checkpoint
import torch.profiler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean

from torch.cuda.amp import autocast

# Enable expandable segments for memroy allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add the parent directory, i.e. transformer, means parent directory of 'scripts' and 'notebooks', to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# Import classes and functions
from scripts.utils import *
from scripts.classes import *
from scripts.utils_wrist import *

def encoder_train(config):
    ### Load configuration parameters
    general = config['general']
    output = config['output']
    params = config['parameters']
    model_name = config['general']['model_name']
    model_family = config['output']['model_family']
    
    checkpoints_folder = config['output']['checkpoints'] + model_family + model_name
    model_summary_folder = config['output']['model_summary'] + model_family + model_name

    print(f"Training model with general settings: {general}", '\n')
    print(f"Training model with parameters: {params}", '\n')
    print(f"Saving files to folders: {checkpoints_folder}", '\n', {model_summary_folder})

    ### Device selction, data loading and preprocessing
    # Select cpu, gpu or mps device for training
    device = select_device()

    # Use a random seed for reproducibility 
    seed = config['general']['random_seed']
    random.seed(seed)

    # Apply the seed globally
    random.seed(seed) # Randomness for CPU devices
    np.random.seed(seed) # Randomness for CPU devices
    torch.manual_seed(seed) # Randomness for CPU and MPS devices
    
    # CUDA device-specific seed handling
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Randomness for CUDA devices

    # Ensure deterministic behavior for CUDA / GPU devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    if config['general']['imu'] == False:
        ### TEST SINGLE ACTION PAIR
        if config['parameters']['subject'] != 'all':
            if config['general']['dataset'] == 'Finger': 
                if config['general']['filter'] == True:
                    df_data = data_loader_filtered()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")

                print(f"Training {config['general']['dataset']} dataset:",df_data.shape)
            
                # Define the subjects and action you want to filter
                selected_subjects = config['parameters']['subject']  # Replace with desired subject IDs
                selected_action = config['parameters']['action'] # Replace with the desired action

                # Filter the DataFrame
                df_custom = df_data[(df_data['subject'].isin(selected_subjects)) & (df_data['action'] == selected_action)]

                # Reset the index of the new DataFrame
                df_custom = df_custom.reset_index(drop=True)
                df_test = df_custom

                # Group data by (subject, actions) pair, normalize it by each (subject, actions) pair and save the scalers or use global normalization
                if config['general']['normalization'] == 'global':
                        df_custom_normalized, scalers = global_normalization(df_custom)
                else:
                    df_custom_normalized, scalers = normalization_group_action(df_custom)


                # Verify the result
                print(df_custom.head())

                # Ratios for train, validation, and test splits
                train_ratio = 0.7
                val_ratio = 0.2
                test_ratio = 0.1

                sequence_length = config['parameters']['sequence_length'] 
                sequence_step_size = config['parameters']['sequence_step_size'] 
                subset = config['parameters']['subset'] 

                # Generate sequences
                x_data, y_data = sequences(df_custom_normalized, sequence_length, sequence_step_size, subset)

                # Calculate sizes for each subset
                total_samples = x_data.size(0)
                train_size = int(train_ratio * total_samples)
                val_size = int(val_ratio * total_samples)
                test_size = total_samples - train_size - val_size  # Remaining samples go to the test set

                # Split the data
                X_train = x_data[:train_size]
                y_train = y_data[:train_size]

                X_val = x_data[train_size:train_size + val_size]
                y_val = y_data[train_size:train_size + val_size]

                X_test = x_data[train_size + val_size:]
                y_test = y_data[train_size + val_size:]

                # Print shapes for verification
                print(f"x_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"x_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                print(f"x_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            elif config['general']['dataset'] == 'Wrist': 
                if config['general']['filter'] == True:
                        df_data = data_loader_filtered_wrist()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original_wrist()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")

                print(f"Training {config['general']['dataset']} dataset:",df_data.shape)
            
                # Define the subjects and action you want to filter
                selected_subjects = config['parameters']['subject']  # Replace with desired subject IDs
                selected_action = config['parameters']['action'] # Replace with the desired action

                # Filter the DataFrame
                df_custom = df_data[(df_data['subject'].isin(selected_subjects)) & (df_data['action'] == selected_action)]

                # Reset the index of the new DataFrame
                df_custom = df_custom.reset_index(drop=True)
                df_test = df_custom

                # Group data by (subject, actions) pair, normalize it by each (subject, actions) pair and save the scalers or use global normalization
                if config['general']['normalization'] == 'global':
                        df_custom_normalized, scalers = global_normalization_wrist(df_custom)
                else:
                    df_custom_normalized, scalers = normalization_group_action_wrist(df_custom)


                # Verify the result
                print(df_custom.head())

                # Ratios for train, validation, and test splits
                train_ratio = 0.7
                val_ratio = 0.2
                test_ratio = 0.1

                sequence_length = config['parameters']['sequence_length'] 
                sequence_step_size = config['parameters']['sequence_step_size'] 
                subset = config['parameters']['subset'] 

                # Generate sequences
                x_data, y_data = sequences_wrist(df_custom_normalized, sequence_length, sequence_step_size, subset)

                # Calculate sizes for each subset
                total_samples = x_data.size(0)
                train_size = int(train_ratio * total_samples)
                val_size = int(val_ratio * total_samples)
                test_size = total_samples - train_size - val_size  # Remaining samples go to the test set

                # Split the data
                X_train = x_data[:train_size]
                y_train = y_data[:train_size]

                X_val = x_data[train_size:train_size + val_size]
                y_val = y_data[train_size:train_size + val_size]

                X_test = x_data[train_size + val_size:]
                y_test = y_data[train_size + val_size:]

                # Print shapes for verification
                print(f"x_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"x_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                print(f"x_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        else:
            ### OUTCOMMENT ALL DATA 
            if config['general']['dataset'] == 'Wrist': 
                if config['general']['filter'] == True:
                    df_data = data_loader_filtered_wrist()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original_wrist()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")
                print(f"Training {config['general']['dataset']} dataset:", df_data.shape)

            elif config['general']['dataset'] == 'Finger':
                # Load the data
                if config['general']['filter'] == True:
                    df_data = data_loader_filtered()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")
                #df_data = data_loader_single(subject=config['parameters']['subject'] , action=config['parameters']['action'] )
                print(f"Training {config['general']['dataset']} dataset:",df_data.shape)

            else:
                print("No valid dataset selected!")
            
            # Check if only one subject or a list of subjects is considered
            if config['parameters']['subject'] != 'all':
                df_data = df_data[df_data['subject'].isin(config['parameters']['subject'])]
                print(f"Training on these subjects {config['parameters']['subject']} and with this shape",df_data.shape)

            # Check if only one action is considered
            if config['parameters']['action'] != 'all':
                df_data = df_data[df_data['action'] == config['parameters']['action']]
                print(f"Training on these actions {config['parameters']['action']} and with this shape",df_data.shape)


            # Group data by (subject, actions) pair, normalize it by each (subject, actions) pair and save the scalers or use global normalization
            if config['general']['dataset'] == 'Wrist': 
                if config['general']['normalization'] == 'global':
                    df_data_normalized, scalers = global_normalization_wrist(df_data)
                else:
                    df_data_normalized, scalers = normalization_group_action_wrist(df_data)
            if config['general']['dataset'] == 'Finger': 
                if config['general']['normalization'] == 'global':
                    df_data_normalized, scalers = global_normalization(df_data)
                else:
                    df_data_normalized, scalers = normalization_group_action(df_data)

            ### Withhold subjects for validation and training
            val_set_subjects = config['general']['validation_set_subjects']
            test_set_subjects = config['general']['test_set_subjects']

            # Split subjects into train, validation, and test sets
            val_subjects = []
            test_subjects = []
            # Extract unique subject IDs from the dataframe
            unique_subjects = [subject for subject in df_data_normalized['subject'].unique() if subject != 7]  # Exclude subject 7

            if config['general']['dataset'] == 'Wrist': 
                while len(test_subjects) < test_set_subjects:
                    ran_sub = random.choice(unique_subjects)
                    if ran_sub not in test_subjects:
                        test_subjects.append(ran_sub)

                while len(val_subjects) < val_set_subjects: 
                    ran_sub = random.choice(unique_subjects)
                    if ran_sub not in test_subjects and ran_sub not in val_subjects:
                        val_subjects.append(ran_sub)

                # Remaining subjects are for training
                train_subjects = [subject for subject in unique_subjects if subject not in test_subjects and subject not in val_subjects]

            elif config['general']['dataset'] == 'Finger': 
                while len(test_subjects) < test_set_subjects:
                    ran_sub = random.randint(1, df_data_normalized['subject'].nunique())
                    if ran_sub not in test_subjects :
                        test_subjects.append(ran_sub)

                while len(val_subjects) < val_set_subjects: 
                    ran_sub = random.randint(1, df_data_normalized['subject'].nunique())
                    if ran_sub not in test_subjects and ran_sub not in val_subjects:
                        val_subjects.append(ran_sub)

                # Remaining subjects are for training
                train_subjects = [i for i in range(1, (1+df_data_normalized['subject'].nunique())) if i not in val_subjects and i not in test_subjects]
            else:
                print("No valid dataset selected!")

            # Sort for consistency
            train_subjects = np.sort(train_subjects)
            val_subjects = np.sort(val_subjects)
            test_subjects = np.sort(test_subjects)
            
            print(f"Training subjects: {train_subjects}")
            print(f"Validation subjects: {val_subjects}")
            print(f"Test subjects: {test_subjects}")
            
            # Separate data into train, validation, and test sets
            train_rows = df_data_normalized[df_data_normalized['subject'].isin(train_subjects)]
            val_rows = df_data_normalized[df_data_normalized['subject'].isin(val_subjects)]
            test_rows = df_data_normalized[df_data_normalized['subject'].isin(test_subjects)]

            # Create and append to respective DataFrames
            df_train = pd.DataFrame()
            df_val = pd.DataFrame()
            df_test = pd.DataFrame()

            df_train = pd.concat([df_train, train_rows], ignore_index=True)
            df_val = pd.concat([df_val, val_rows], ignore_index=True)
            df_test = pd.concat([df_test, test_rows], ignore_index=True)

            # Reshape for sequence input, adjust stepsize and subset
            sequence_length = config['parameters']['sequence_length'] 
            sequence_step_size = config['parameters']['sequence_step_size'] 
            subset = config['parameters']['subset'] 


            # Preprocess data for train, validation, and test sets
            if config['general']['dataset'] == 'Wrist': 
                X_train, y_train = sequences_wrist(df_train, sequence_length, sequence_step_size, subset)
                X_val, y_val = sequences_wrist(df_val, sequence_length, sequence_step_size, subset)
                X_test, y_test = sequences_wrist(df_test, sequence_length, sequence_step_size, subset)
            else:
                X_train, y_train = sequences(df_train, sequence_length, sequence_step_size, subset)
                X_val, y_val = sequences(df_val, sequence_length, sequence_step_size, subset)
                X_test, y_test = sequences(df_test, sequence_length, sequence_step_size, subset)

            # Print shapes for verification
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    elif config['general']['imu'] == True:
        ### TEST SINGLE ACTION PAIR
        if config['parameters']['subject'] != 'all':
            if config['general']['dataset'] == 'Finger': 
                if config['general']['filter'] == True:
                    df_data = data_loader_filtered_imu()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")

                print(f"Training {config['general']['dataset']} dataset:",df_data.shape)
            
                # Define the subjects and action you want to filter
                selected_subjects = config['parameters']['subject']  # Replace with desired subject IDs
                selected_action = config['parameters']['action'] # Replace with the desired action

                # Filter the DataFrame
                df_custom = df_data[(df_data['subject'].isin(selected_subjects)) & (df_data['action'] == selected_action)]

                # Reset the index of the new DataFrame
                df_custom = df_custom.reset_index(drop=True)
                df_test = df_custom

                # Group data by (subject, actions) pair, normalize it by each (subject, actions) pair and save the scalers or use global normalization
                if config['general']['normalization'] == 'global':
                        df_custom_normalized, scalers = global_normalization(df_custom)
                else:
                    df_custom_normalized, scalers = normalization_group_action_imu(df_custom)


                # Verify the result
                print(df_custom.head())

                # Ratios for train, validation, and test splits
                train_ratio = 0.7
                val_ratio = 0.2
                test_ratio = 0.1

                sequence_length = config['parameters']['sequence_length'] 
                sequence_step_size = config['parameters']['sequence_step_size'] 
                subset = config['parameters']['subset'] 

                # Generate sequences
                x_data, y_data = sequences_imu(df_custom_normalized, sequence_length, sequence_step_size, subset)

                # Calculate sizes for each subset
                total_samples = x_data.size(0)
                train_size = int(train_ratio * total_samples)
                val_size = int(val_ratio * total_samples)
                test_size = total_samples - train_size - val_size  # Remaining samples go to the test set

                # Split the data
                X_train = x_data[:train_size]
                y_train = y_data[:train_size]

                X_val = x_data[train_size:train_size + val_size]
                y_val = y_data[train_size:train_size + val_size]

                X_test = x_data[train_size + val_size:]
                y_test = y_data[train_size + val_size:]

                # Print shapes for verification
                print(f"x_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"x_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                print(f"x_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            elif config['general']['dataset'] == 'Wrist': 
                if config['general']['filter'] == True:
                        df_data = data_loader_filtered_wrist_imu()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original_wrist()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")

                print(f"Training {config['general']['dataset']} dataset:",df_data.shape)
            
                # Define the subjects and action you want to filter
                selected_subjects = config['parameters']['subject']  # Replace with desired subject IDs
                selected_action = config['parameters']['action'] # Replace with the desired action

                # Filter the DataFrame
                df_custom = df_data[(df_data['subject'].isin(selected_subjects)) & (df_data['action'] == selected_action)]

                # Reset the index of the new DataFrame
                df_custom = df_custom.reset_index(drop=True)
                df_test = df_custom

                # Group data by (subject, actions) pair, normalize it by each (subject, actions) pair and save the scalers or use global normalization
                if config['general']['normalization'] == 'global':
                        df_custom_normalized, scalers = global_normalization_wrist(df_custom)
                else:
                    df_custom_normalized, scalers = normalization_group_action_wrist_imu(df_custom)


                # Verify the result
                print(df_custom.head())

                # Ratios for train, validation, and test splits
                train_ratio = 0.7
                val_ratio = 0.2
                test_ratio = 0.1

                sequence_length = config['parameters']['sequence_length'] 
                sequence_step_size = config['parameters']['sequence_step_size'] 
                subset = config['parameters']['subset'] 

                # Generate sequences
                x_data, y_data = sequences_wrist_imu(df_custom_normalized, sequence_length, sequence_step_size, subset)

                # Calculate sizes for each subset
                total_samples = x_data.size(0)
                train_size = int(train_ratio * total_samples)
                val_size = int(val_ratio * total_samples)
                test_size = total_samples - train_size - val_size  # Remaining samples go to the test set

                # Split the data
                X_train = x_data[:train_size]
                y_train = y_data[:train_size]

                X_val = x_data[train_size:train_size + val_size]
                y_val = y_data[train_size:train_size + val_size]

                X_test = x_data[train_size + val_size:]
                y_test = y_data[train_size + val_size:]

                # Print shapes for verification
                print(f"x_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                print(f"x_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
                print(f"x_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        else:
            ### OUTCOMMENT ALL DATA 
            if config['general']['dataset'] == 'Wrist': 
                if config['general']['filter'] == True:
                    df_data = data_loader_filtered_wrist_imu()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original_wrist()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")
                print(f"Training {config['general']['dataset']} dataset:", df_data.shape)

            elif config['general']['dataset'] == 'Finger':
                # Load the data
                if config['general']['filter'] == True:
                    df_data = data_loader_filtered_imu()
                elif config['general']['filter'] == False:
                    df_data = data_loader_original()
                else:
                    print("Filter type not set! Set filter variable in config.yaml file.")
                #df_data = data_loader_single(subject=config['parameters']['subject'] , action=config['parameters']['action'] )
                print(f"Training {config['general']['dataset']} dataset:",df_data.shape)

            else:
                print("No valid dataset selected!")
            
            # Check if only one subject or a list of subjects is considered
            if config['parameters']['subject'] != 'all':
                df_data = df_data[df_data['subject'].isin(config['parameters']['subject'])]
                print(f"Training on these subjects {config['parameters']['subject']} and with this shape",df_data.shape)

            # Check if only one action is considered
            if config['parameters']['action'] != 'all':
                df_data = df_data[df_data['action'] == config['parameters']['action']]
                print(f"Training on these actions {config['parameters']['action']} and with this shape",df_data.shape)


            # Group data by (subject, actions) pair, normalize it by each (subject, actions) pair and save the scalers or use global normalization
            if config['general']['dataset'] == 'Wrist': 
                if config['general']['normalization'] == 'global':
                    df_data_normalized, scalers = global_normalization_wrist(df_data)
                else:
                    df_data_normalized, scalers = normalization_group_action_wrist_imu(df_data)
            if config['general']['dataset'] == 'Finger': 
                if config['general']['normalization'] == 'global':
                    df_data_normalized, scalers = global_normalization(df_data)
                else:
                    df_data_normalized, scalers = normalization_group_action_imu(df_data)

            ### Withhold subjects for validation and training
            val_set_subjects = config['general']['validation_set_subjects']
            test_set_subjects = config['general']['test_set_subjects']

            # Split subjects into train, validation, and test sets
            val_subjects = []
            test_subjects = []
            # Extract unique subject IDs from the dataframe
            unique_subjects = [subject for subject in df_data_normalized['subject'].unique() if subject != 7]  # Exclude subject 7

            if config['general']['dataset'] == 'Wrist': 
                while len(test_subjects) < test_set_subjects:
                    ran_sub = random.choice(unique_subjects)
                    if ran_sub not in test_subjects:
                        test_subjects.append(ran_sub)

                while len(val_subjects) < val_set_subjects: 
                    ran_sub = random.choice(unique_subjects)
                    if ran_sub not in test_subjects and ran_sub not in val_subjects:
                        val_subjects.append(ran_sub)

                # Remaining subjects are for training
                train_subjects = [subject for subject in unique_subjects if subject not in test_subjects and subject not in val_subjects]

            elif config['general']['dataset'] == 'Finger': 
                while len(test_subjects) < test_set_subjects:
                    ran_sub = random.randint(1, df_data_normalized['subject'].nunique())
                    if ran_sub not in test_subjects :
                        test_subjects.append(ran_sub)

                while len(val_subjects) < val_set_subjects: 
                    ran_sub = random.randint(1, df_data_normalized['subject'].nunique())
                    if ran_sub not in test_subjects and ran_sub not in val_subjects:
                        val_subjects.append(ran_sub)

                # Remaining subjects are for training
                train_subjects = [i for i in range(1, (1+df_data_normalized['subject'].nunique())) if i not in val_subjects and i not in test_subjects]
            else:
                print("No valid dataset selected!")

            # Sort for consistency
            train_subjects = np.sort(train_subjects)
            val_subjects = np.sort(val_subjects)
            test_subjects = np.sort(test_subjects)
            
            print(f"Training subjects: {train_subjects}")
            print(f"Validation subjects: {val_subjects}")
            print(f"Test subjects: {test_subjects}")
            
            # Separate data into train, validation, and test sets
            train_rows = df_data_normalized[df_data_normalized['subject'].isin(train_subjects)]
            val_rows = df_data_normalized[df_data_normalized['subject'].isin(val_subjects)]
            test_rows = df_data_normalized[df_data_normalized['subject'].isin(test_subjects)]

            # Create and append to respective DataFrames
            df_train = pd.DataFrame()
            df_val = pd.DataFrame()
            df_test = pd.DataFrame()

            df_train = pd.concat([df_train, train_rows], ignore_index=True)
            df_val = pd.concat([df_val, val_rows], ignore_index=True)
            df_test = pd.concat([df_test, test_rows], ignore_index=True)

            # Reshape for sequence input, adjust stepsize and subset
            sequence_length = config['parameters']['sequence_length'] 
            sequence_step_size = config['parameters']['sequence_step_size'] 
            subset = config['parameters']['subset'] 


            # Preprocess data for train, validation, and test sets
            if config['general']['dataset'] == 'Wrist': 
                X_train, y_train = sequences_wrist_imu(df_train, sequence_length, sequence_step_size, subset)
                X_val, y_val = sequences_wrist_imu(df_val, sequence_length, sequence_step_size, subset)
                X_test, y_test = sequences_wrist_imu(df_test, sequence_length, sequence_step_size, subset)
            else:
                X_train, y_train = sequences_imu(df_train, sequence_length, sequence_step_size, subset)
                X_val, y_val = sequences_imu(df_val, sequence_length, sequence_step_size, subset)
                X_test, y_test = sequences_imu(df_test, sequence_length, sequence_step_size, subset)

            # Print shapes for verification
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    else:
        print("Choose if IMU data should be used or not!")

    # Model initialization 
    d_model = config['parameters']['d_model']   # Embedding dimension
    input_dim = config['parameters']['input_dim']   # 3 PPG signals (red, green, IR)
    output_dim = config['parameters']['output_dim']  # 1 ECG target per time step
    nhead = config['parameters']['nhead']   # Attention heads
    num_layers = config['parameters']['num_layers']   # Number of transformer layers
    batch_size = config['parameters']['batch_size']   # Batch size
    dropout = config['parameters']['dropout']   # Dropout

    
    # Convert tensors to Datasets
    train_dataset = PreprocessedDataset(X_train, y_train)
    val_dataset = PreprocessedDataset(X_val, y_val)
    
    # Create DataLoaders with a reproducible generator
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=config["general"]["train_shuffling"], num_workers=0, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize the Transformer model
    model = Point2PointEncoderTransformer(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout).to(device) 
    
    # Save the summary of the model
    #save_model_summary(model, X_train, y_train, device, model_name, batch_size, model_summary_folder)

    # Wrap with checkpointing
    #model = checkpoint(model)

    # Loss function: Mean Squared Error for regression tasks
    loss_fn = nn.MSELoss()

    # Optimizer: AdamW optimizer
    optimizer = optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,               # Lower learning rate for better control in transformers
        betas=(0.9, 0.999),    # Default Adam settings
        eps=1e-08, 
        weight_decay=0.01,     # Regularization to prevent overfitting
        amsgrad=False          # Default is suitable for most use cases
    )

    # Number of epochs to train
    num_epochs = config['parameters']['num_epochs']   

    # Initialize a learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Clear any residual memory before training
    torch.cuda.empty_cache()

    # Arrays for storing losses and epochs
    training_loss = np.array([])
    validation_loss = np.array([])
    epochs = np.array([])
    best_models = np.array([])

    # Early stopping and checkpoint parameters
    patience = 10
    min_delta = 1e-4
    best_val_loss = float('inf')
    early_stop_counter = 0


    ### Training
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0 # Initialize running loss
        # Iterate through the batches in the train_loader to load the data in batches
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass through the model
            predictions = model(batch_X)

            # Calculate loss (MSE between predicted ECG and actual ECG)
            loss = loss_fn(predictions, batch_y)

            # Backward pass (compute gradients)
            loss.backward()

            # Update the weights
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * batch_X.size(0)
        

        # Calculate the average loss for the epoch
        avg_train_loss = running_loss / len(X_train)
        train_rmse = torch.sqrt(torch.tensor(avg_train_loss)) # MSE needs to be calculated at the end of each batch, scaled by batch size and the RMSE should calculated at the end of the epoch (metric)
        training_loss = np.append(training_loss, train_rmse.cpu())

        print(f"Training of epoch {epoch+1} done, starting validation!")
        # Validation metrics with batching
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            # Iterate through the batches in the val_loader to load the data in batches
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val = batch_X_val.to(device)
                batch_y_val = batch_y_val.to(device)
            
                # Forward pass
                val_predictions = model(batch_X_val)

                # Calculate loss for this batch
                val_loss = loss_fn(val_predictions, batch_y_val)

                # Accumulate total validation loss
                total_val_loss += val_loss.item() * batch_X_val.size(0)  # Weighted by batch size


            # Average validation loss over all samples
            avg_val_loss = total_val_loss / len(X_val) 
            val_rmse = torch.sqrt(torch.tensor(avg_val_loss)) # MSE needs to be calculated at the end of each batch, scaled by batch size and the RMSE should calculated at the end of the epoch (metric)
            validation_loss = np.append(validation_loss, val_rmse.cpu())

            # Step the learning rate scheduler with the validation loss
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                # Save checkpoint
                checkpoint_path = f"{checkpoints_folder}/epoch{epoch+1}.pth"
                save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, checkpoint_path)

                # Save the model
                torch.save(model.state_dict(), f"../models/{model_family}{model_name}_trained_model_epoch{epoch+1}.pth")

                # Save the epoch of this best model
                best_models = np.append(best_models, int(epoch+1))

                epochs = np.append(epochs, int(epoch+1))
                print(f"Checkpoint saved at epoch {epoch + 1}.")
            else:
                epochs = np.append(epochs, int(epoch+1))
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            #print(f"Memory usage: {psutil.virtual_memory().percent}%")
            print(f"Epoch {epoch + 1}/{num_epochs} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Clear any residual memory before start of new epoch
            torch.cuda.empty_cache()
    print("Saving X_test, y_test with shape:", X_test.shape, y_test.shape)
    # Save training, validation, and test data to a pickle file
    metadata_to_save = {
        "Configuration_file": config, # Safe the entire config setting
        # "Training subjects": train_subjects, # Save the subjects used for training
        # "Validation subjects": val_subjects, # Save the subjects used for validation
        # "Test subjects": test_subjects, # Save the subjects used for testing
        "Scalers": scalers,  # Dictionary with all the scalers
        # "X_train": X_train,  # PyTorch tensor
        # "y_train": y_train,  # PyTorch tensor
        # "X_val": X_val,      # PyTorch tensor
        # "y_val": y_val,      # PyTorch tensor
        "X_test": X_test,    # PyTorch tensor
        "y_test": y_test,    # PyTorch tensor
        # "df_train": df_train,  # Pandas DataFrame
        # "df_val": df_val,      # Pandas DataFrame
        #"df_test": df_test,     # Pandas DataFrame
        "Training_loss": training_loss, # Training loss
        "Validation_loss": validation_loss, # Validation loss
        "Epochs": epochs, # Epochs
        "Best models": best_models # Epochs of the best models
    }

    metadata_to_save = load_data_to_device(metadata_to_save, 'cpu')
    # Save to a pickle file
    with open(f"{model_summary_folder}/{model_name}_saved_metadata.pkl", "wb") as f:
        pickle.dump(metadata_to_save, f)

    print("Model trained successful!")