import os
import sys
import psutil
import random
import torch
import pickle
import yaml
import numpy as np
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean

# Add the parent directory (i.e. transformer, means parent directory of 'scripts' and 'notebooks') to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# Import the function
from scripts.utils import *
from scripts.classes import *
from scripts.figure_layout import *

def encoder_eval(config):
    # Get the model name of the trained model
    model_name = config['general']['model_name']
    model_family = config['output']['model_family']
    model_summary_folder = config['output']['model_summary'] + model_family + model_name
   
    # Get cpu, gpu or mps device for training.
    device = select_device()

    # Load data
    with open(f"{model_summary_folder}/{model_name}_saved_metadata.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    # Convert loaded data to device
    loaded_data = load_data_to_device(loaded_data, device)

    # Retrieve data
    loaded_config = loaded_data["Configuration_file"]
    scalers = loaded_data["Scalers"] # Dictionary
    # train_subjects = loaded_data["Training subjects"]
    # val_subjects = loaded_data["Validation subjects"]
    # test_subjects = loaded_data["Test subjects"]
    # X_train = loaded_data["X_train"]  # PyTorch tensor
    # y_train = loaded_data["y_train"]  # PyTorch tensor
    # X_val = loaded_data["X_val"]      # PyTorch tensor
    # y_val = loaded_data["y_val"]      # PyTorch tensor
    X_test = loaded_data["X_test"]      # PyTorch tensor
    y_test = loaded_data["y_test"]      # PyTorch tensor
    # df_train = loaded_data["df_train"]  # Pandas DataFrame
    # df_val = loaded_data["df_val"]      # Pandas DataFrame
    #df_test = loaded_data["df_test"] # Pandas DataFrame
    training_loss = loaded_data["Training_loss"]  # Numpy array
    validation_loss = loaded_data["Validation_loss"]  # Numpy array
    epochs = loaded_data["Epochs"]  # Numpy array
    best_models = loaded_data[ "Best models"] # Numpy array

    print("This is X_Test shape", X_test.shape)

    assert loaded_config['general']['model_name'] == model_name
    general = loaded_config['general']
    params = loaded_config['parameters']
    assert loaded_config['output']['model_family'] == model_family 
    results_folder = loaded_config['output']['results'] + model_family + model_name 
    d_model = loaded_config['parameters']['d_model']   # Embedding dimension
    input_dim = loaded_config['parameters']['input_dim']   # 3 PPG signals (red, green, IR)
    output_dim = loaded_config['parameters']['output_dim']  # 1 ECG target per time step
    nhead = loaded_config['parameters']['nhead']   # Attention heads
    num_layers = loaded_config['parameters']['num_layers']   # Number of transformer layers
    batch_size = loaded_config['parameters']['batch_size']   # Batch size
    sequence_length = loaded_config['parameters']['sequence_length'] 
    num_epochs = loaded_config['parameters']['num_epochs']  
    use_dataloader = loaded_config['general']['use_dataloader'] # Use a dataloader or not
    random_seed = loaded_config['general']['random_seed']

    print(f"Evaluating model with general settings: {general}", '\n')
    print(f"Evaluating model with parameters: {params}", '\n')
    print(f"Saving files to folders: {results_folder}")

    # Use a random seed for reproducibility 
    seed = config['general']['random_seed']
    random.seed(seed)
    np.random.seed(seed) 
   
    # Initialize the Transformer model
    model = Point2PointEncoderTransformer(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device) 

    # Load the model
    model.load_state_dict(torch.load(f"../models/{model_family}{model_name}_trained_model_epoch{int(best_models[-1])}.pth", map_location=torch.device(device)))


    ### Validation
    # Initialize storage for aggregated predictions and actual values
    ecg_predictions = []
    ecg_actuals = []
    ppg = []
    # subjects = []  # Store subject info for each batch
    # actions = []   # Store action info for each batch

    # Loss function: Mean Squared Error for regression tasks
    loss_fn = nn.MSELoss()

    # Test Loss
    test_loss = np.array([])
    running_test_loss = 0

    # Convert tensors to Datasets
    test_dataset = PreprocessedDataset(X_test, y_test)
    # Create DataLoaders for each set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Iterate over the validation set in batches
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        # Iterate through the batches in the test_loader to load the data in batches
        for batch_idx, (batch_X_test, batch_y_test) in enumerate(test_loader):
            # Move the batch data to the device (GPU or CPU)
            batch_X_test = batch_X_test.to(device)
            batch_y_test = batch_y_test.to(device)
            
            # # Get the start and end index of the current batch in df_test
            # start_idx = batch_idx * batch_size
            # end_idx = start_idx + len(batch_X_test)
            
            # # Retrieve the corresponding (subject, action) pair for this batch from df_test
            # batch_subjects = df_test.iloc[start_idx:end_idx]['subject'].values
            # batch_actions = df_test.iloc[start_idx:end_idx]['action'].values

            # Forward pass to get predictions
            batch_predictions = model(batch_X_test)

            # Calculate loss for this batch
            loss = loss_fn(batch_predictions, batch_y_test)

            # Accumulate total validation loss
            running_test_loss += loss.item() * batch_X_test.size(0)

            # Store predictions, actuals, subjects, and actions
            ecg_predictions.append(batch_predictions.cpu())  # Move to CPU for numpy/scaler operations
            ecg_actuals.append(batch_y_test.cpu())
            ppg.append(batch_X_test.cpu())
            # subjects.extend(batch_subjects)
            # actions.extend(batch_actions)

    # Average the test loss over all samples
    avg_test_loss = running_test_loss / len(X_test)
    test_rmse = torch.sqrt(torch.tensor(avg_test_loss))
    test_loss = np.append(test_loss, test_rmse.cpu())

    # Concatenate all batches
    ecg_predictions = torch.cat(ecg_predictions, dim=0)
    ecg_actuals = torch.cat(ecg_actuals, dim=0)
    ppg = torch.cat(ppg, dim=0)

    # # Initialize lists for original scale data
    # ecg_predictions_original_scale = []
    # ecg_actuals_original_scale = []
    # ppg_original_scale = []

    # # Process each sequence
    # for i in range(len(ecg_predictions)):
    #     # Get subject and action for the current sequence
    #     subject = subjects[i]
    #     action = actions[i]

    #     # Retrieve the correct scalers
    #     scaler_input = scalers[(subject, action)]['input_scaler']
    #     scaler_target = scalers[(subject, action)]['target_scaler']

    #     # Inverse transform predictions and actuals for the current sequence
    #     ecg_pred = ecg_predictions[i].squeeze(-1).numpy()  # Shape: [sequence_length]
    #     ecg_act = ecg_actuals[i].squeeze(-1).numpy()       # Shape: [sequence_length]
    #     ppg_seq = ppg[i].numpy()                          # Shape: [sequence_length, 3]

    #     ecg_predictions_original_scale.append(scaler_target.inverse_transform(ecg_pred.reshape(-1, 1)).flatten())
    #     ecg_actuals_original_scale.append(scaler_target.inverse_transform(ecg_act.reshape(-1, 1)).flatten())
    #     ppg_original_scale.append(scaler_input.inverse_transform(ppg_seq))

    # # Convert back to arrays
    # ecg_predictions_original_scale = np.array(ecg_predictions_original_scale)
    # ecg_actuals_original_scale = np.array(ecg_actuals_original_scale)
    # ppg_original_scale = np.array(ppg_original_scale)

    # # Separate PPG channels 
    # red_ppg = ppg_original_scale[:, :, 0]  # Red PPG
    # ir_ppg = ppg_original_scale[:, :, 1]   # IR PPG
    # green_ppg = ppg_original_scale[:, :, 2]  # Green PPG

    if config['general']['train'] == False:
        ### Normalized Evaluation metrics
        # Predictions and actual values (normalized and flattened)
        ecg_predictions_arr = np.array(ecg_predictions).flatten()
        ecg_actuals_arr = np.array(ecg_actuals).flatten()

        # Calculate the range of the actual data for normalization
        actual_range_normalized = np.ptp(ecg_actuals)  # Peak-to-peak (max - min)

        # Euclidean Distance
        euclidean_distance_normalized = euclidean(ecg_predictions_arr, ecg_actuals_arr)

        # Dynamic Time Warping (DTW)
        downsampling_factor_dtw = 10
        batch_size_dtw = 10 
        dtw_distance_normalized = compute_batched_dtw(ecg_predictions_arr, ecg_actuals_arr, batch_size_dtw, downsampling_factor_dtw)
        # dtw_distance = alignment.distance

        # Pearson Correlation
        pearson_corr_normalized, _ = pearsonr(ecg_predictions_arr, ecg_actuals_arr)

        # Spearman Correlation
        spearman_corr_normalized, _ = spearmanr(ecg_predictions_arr, ecg_actuals_arr)

        # Mean Squared Error (MSE)
        mse_normalized = np.mean((ecg_predictions_arr - ecg_actuals_arr) ** 2)

        # Mean Absolute Error (MAE)
        mae_normalized = np.mean(np.abs(ecg_predictions_arr - ecg_actuals_arr))

        # Root Mean Squared Error (RMSE)
        rmse_normalized = np.sqrt(mse_normalized)

        # Normalized Root Mean Squared Error (NRMSE)
        nrmse_normalized = rmse_normalized / actual_range_normalized

        # Normalized Mean Absolute Error (NMAE)
        nmae_normalized = mae_normalized / actual_range_normalized

        # Print metrics
        metrics_normalized = {
            "Training_loss": training_loss,
            "Validation_loss": validation_loss,
            "Test_loss": test_loss,
            "Epochs": epochs, 
            "Euclidean Distance": euclidean_distance_normalized,
            "DTW Distance": dtw_distance_normalized,
            "Pearson Correlation": pearson_corr_normalized,
            "Spearman Correlation": spearman_corr_normalized,
            "MSE": mse_normalized,
            "MAE": mae_normalized,
            "RMSE": rmse_normalized,
            "NRMSE": nrmse_normalized,
            "NMAE": nmae_normalized,
            "Parameters": config['parameters'],  # Add config file entries
            "General": config['general'],
            "Output": config['output'],
        }

        for metric, value in metrics_normalized.items():
            #print(f"{metric}: {value:.4f}")
            print(f"{metric}: {value}")

        # Write metrics to a file
        metrics_file = os.path.join(results_folder, f"{model_name}_metrics.txt")
        with open(metrics_file, "w") as f:
            for metric, value in metrics_normalized.items():
                #f.write(f"{metric}: {value:.4f}\n")
                f.write(f"{metric}: {value}\n")

        print(f"Metrics written to {metrics_file}")



        ### Plots
        # Call a plot(...) method to create them
        # Randomly select an index from the validation data
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, training_loss,  label='Training Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')
        plt.yscale('log')
        plt.title(f"Training and Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        #plt.xticks(epochs)
        plt.legend()

        plt.savefig(f"{results_folder}/{model_name}_loss_functions.png")

            # Repeat test loss across all epochs for visualization
        test_losses = [test_loss] * len(epochs)

        # Plot the losses
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, training_loss, label="Training Loss", marker='o')
        plt.plot(epochs, validation_loss, label="Validation Loss", marker='o')
        plt.plot(epochs, test_losses, label="Test Loss", linestyle='--', color='red')

        # Add labels, title, legend
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training, Validation, and Test Loss")
        plt.legend()

        plt.savefig(f"{results_folder}/{model_name}_test_loss.png")


        ecg_predictions_reshaped = ecg_predictions.squeeze()
        ecg_predictions_reshaped_array = np.array(ecg_predictions_reshaped)
        ecg_actuals_reshaped = ecg_actuals.squeeze()
        ecg_actuals_reshaped_array = np.array(ecg_actuals_reshaped)

        random_index = np.random.randint(0, len(ecg_predictions_reshaped_array))
        ppg_scaling_factor = config['parameters']['ppg_scaling_factor']

        # Select the corresponding actual and predicted ECG signals
        ecg_predictions_random = ecg_predictions_reshaped_array[random_index]  # Predicted ECG signal
        ecg_actuals_random = ecg_actuals_reshaped_array[random_index]  # Actual ECG signal

        # Set the opacity value of alpha for the ppg signals
        alpha = 0.3

        # Plot the actual and predicted ECG
        plt.figure(figsize=(10, 5))
        plt.plot(ecg_actuals_random, label='Actual ECG')
        plt.plot(ecg_predictions_random, label='Predicted ECG')
        plt.title(f"ECG Prediction vs Actual (Sequence {random_index})")
        plt.xlabel('Time Step')
        plt.ylabel('ECG Signal')
        plt.legend()

        plt.savefig(f"{results_folder}/{model_name}_random_seq.png")

        # Plot the actual and predicted ECG with the input ppg signals
        plt.figure(figsize=(10, 5))
        plt.plot(ecg_actuals_random, label='Actual ECG')
        plt.plot(ecg_predictions_random, label='Predicted ECG')
        # plt.plot(ppg_scaling_factor*red_ppg[random_index], label="Red PPG", alpha=alpha)
        # plt.plot(ppg_scaling_factor*ir_ppg[random_index], label="IR PPG", alpha=alpha)
        # plt.plot(ppg_scaling_factor*green_ppg[random_index], label="Green PPG", alpha=alpha)
        plt.title(f"ECG Prediction vs Actual (Sequence {random_index}) with PPG signals")
        plt.xlabel('Time Step')
        plt.ylabel('ECG Signal')
        plt.legend()

        plt.savefig(f"{results_folder}/{model_name}_random_seq_ppg.png")

        print("Evaluation finished!")

    # Finger but not IMU
    if (config['general']['train'] == False) and (loaded_config['general']['dataset']=='Finger') and (config['general']['imu']==False):
        ecg_predictions_reshaped = ecg_predictions.squeeze()
        ecg_predictions_reshaped_array = np.array(ecg_predictions_reshaped)
        ecg_actuals_reshaped = ecg_actuals.squeeze()
        ecg_actuals_reshaped_array = np.array(ecg_actuals_reshaped)
        input_reshaped = ppg.squeeze()
        input_reshaped_array = np.array(input_reshaped)

        random_index = np.random.randint(0, len(ecg_predictions_reshaped_array))
        ppg_scaling_factor = config['parameters']['ppg_scaling_factor']

        # Select the corresponding actual and predicted ECG signals
        ecg_predictions_random = ecg_predictions_reshaped_array[random_index]  # Predicted ECG signal
        ecg_actuals_random = ecg_actuals_reshaped_array[random_index]  # Actual ECG signal
        input_random = input_reshaped_array[random_index]
    
        fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

        sampling_frequency = 500
        time = np.arange(0, sequence_length) / sampling_frequency

        print(input_random)
        print(input_random.shape)
        print(type(input_random))

        red_ppg = input_random[:, 0]   # R PPG
        ir_ppg = input_random[:, 1]    # IR PPG
        green_ppg = input_random[:, 2] # G PPG

        # Plot for Green PPG signals while sitting
        axes[0].plot(
            time,
            ecg_actuals_random,
            label='Actual ECG',
            #color='g'
        )
        axes[0].plot(
            time,
            ecg_predictions_random,
            label='Predicted ECG',
            #color='g'
        )
        axes[0].set_xlabel("Time in seconds")
        axes[0].set_ylabel("Normalized amplitude")
        axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

        # Plot for Green PPG signals while running
        axes[1].plot(
            time,
            green_ppg,
            label='Green PPG',
            color='g'
        )
        axes[1].plot(
            time,
            ir_ppg,
            label='IR PPG',
            color='darkviolet'
        )
        axes[1].plot(
            time,
            red_ppg,
            label='Red PPG',
            color='r'
        )
        axes[1].set_xlabel("Time in seconds")
        axes[1].set_ylabel("Normalized amplitude")
        axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        plt.savefig(f"../img/finger_predictions/{model_name}_ecg.pdf")


        print("Evaluation finished!")

    # Finger and IMU
    elif (config['general']['train'] == False) and (loaded_config['general']['dataset']=='Finger') and (loaded_config['general']['imu']==True):
            ecg_predictions_reshaped = ecg_predictions.squeeze()
            ecg_predictions_reshaped_array = np.array(ecg_predictions_reshaped)
            ecg_actuals_reshaped = ecg_actuals.squeeze()
            ecg_actuals_reshaped_array = np.array(ecg_actuals_reshaped)
            print("PPG shape", ppg.shape)
            input_reshaped = ppg.squeeze()
            input_reshaped_array = np.array(input_reshaped)

            random_index = np.random.randint(0, len(ecg_predictions_reshaped_array))
            print(ecg_predictions_reshaped_array.shape)
            #random_index = 2
            print("random index", random_index)
            ppg_scaling_factor = config['parameters']['ppg_scaling_factor']
            print("shapes", ecg_predictions_reshaped.shape)

            # Select the corresponding actual and predicted ECG signals
            ecg_predictions_random = ecg_predictions_reshaped_array[random_index]  # Predicted ECG signal
            ecg_actuals_random = ecg_actuals_reshaped_array[random_index]  # Actual ECG signal
            input_random = input_reshaped_array[random_index]
        
            fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

            sampling_frequency = 500
            time = np.arange(0, sequence_length) / sampling_frequency

            print("hey")

            red_ppg = input_random[:, 0]   # R PPG
            ir_ppg = input_random[:, 1]    # IR PPG
            green_ppg = input_random[:, 2] # G PPG
            a_x = input_random[:, 3]
            a_y = input_random[:, 4]
            a_z = input_random[:, 5]
            g_x = input_random[:, 6]
            g_y = input_random[:, 7]
            g_z = input_random[:, 8]

            # Plot for Green PPG signals while sitting
            axes[0].plot(
                time,
                ecg_actuals_random,
                label='Actual ECG',
                #color='g'
            )
            axes[0].plot(
                time,
                ecg_predictions_random,
                label='Predicted ECG',
                #color='g'
            )
            axes[0].set_xlabel("Time in seconds")
            axes[0].set_ylabel("Normalized amplitude")
            axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

            # Plot for Green PPG signals while running
            axes[1].plot(
                time,
                green_ppg,
                label='Green PPG',
                color='g'
            )
            axes[1].plot(
                time,
                ir_ppg,
                label='IR PPG',
                color='darkviolet'
            )
            axes[1].plot(
                time,
                red_ppg,
                label='Red PPG',
                color='r'
            )
            axes[1].set_xlabel("Time in seconds")
            axes[1].set_ylabel("Normalized amplitude")
            axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

            plt.savefig(f"../img/finger_predictions/{model_name}_ecg.pdf")

            fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns
            # Plot for Green PPG signals while sitting
            axes[0].plot(
                time,
                a_x,
                label='a_x',
                color='#4b0082'
            )
            axes[0].plot(
                time,
                a_y,
                label='a_y',
                color='#8b4513'
            )
            axes[0].plot(
                time,
                a_z,
                label='a_z',
                color='#808000'
            )
            axes[0].set_xlabel("Time in seconds")
            axes[0].set_ylabel("Normalized amplitude")
            axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

            # Plot for Green PPG signals while running
            axes[1].plot(
                time,
                g_x,
                label='g_x',
                color='c'
            )
            axes[1].plot(
                time,
                g_y,
                label='g_y',
                color='m'
            )
            axes[1].plot(
                time,
                g_z,
                label='g_z',
                color='y'
            )
            axes[1].set_xlabel("Time in seconds")
            axes[1].set_ylabel("Normalized amplitude")
            axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

            plt.savefig(f"../img/finger_predictions/{model_name}_imu.pdf")

            # Create 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 14))  # 2 rows, 2 columns

            # Plot ECG signals on the top left
            axes[0, 0].plot(time, ecg_actuals_random, label='Actual ECG')
            axes[0, 0].plot(time, ecg_predictions_random, label='Predicted ECG')
            axes[0, 0].set_xlabel("Time in seconds")
            axes[0, 0].set_ylabel("Normalized amplitude")
            axes[0, 0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

            # Plot PPG signals on the top right
            axes[0, 1].plot(time, green_ppg, label='Green PPG', color='g')
            axes[0, 1].plot(time, ir_ppg, label='IR PPG', color='darkviolet')
            axes[0, 1].plot(time, red_ppg, label='Red PPG', color='r')
            axes[0, 1].set_xlabel("Time in seconds")
            axes[0, 1].set_ylabel("Normalized amplitude")
            axes[0, 1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

            # Plot accelerometer signals on the bottom left
            axes[1, 0].plot(time, a_x, label='a_x', color='#4b0082')
            axes[1, 0].plot(time, a_y, label='a_y', color='#8b4513')
            axes[1, 0].plot(time, a_z, label='a_z', color='#808000')
            axes[1, 0].set_xlabel("Time in seconds")
            axes[1, 0].set_ylabel("Normalized amplitude")
            axes[1, 0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

            # Plot gyroscope signals on the bottom right
            axes[1, 1].plot(time, g_x, label='g_x', color='c')
            axes[1, 1].plot(time, g_y, label='g_y', color='m')
            axes[1, 1].plot(time, g_z, label='g_z', color='y')
            axes[1, 1].set_xlabel("Time in seconds")
            axes[1, 1].set_ylabel("Normalized amplitude")
            axes[1, 1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig(f"../img/finger_predictions/{model_name}_ecg_ppg_acc_gyro.pdf")

            print("Evaluation finished!")

    # Wrist but no IMU
    elif (config['general']['train'] == False) and (loaded_config['general']['dataset']=='Wrist') and (config['general']['imu']==False):
        ecg_predictions_reshaped = ecg_predictions.squeeze()
        ecg_predictions_reshaped_array = np.array(ecg_predictions_reshaped)
        ecg_actuals_reshaped = ecg_actuals.squeeze()
        ecg_actuals_reshaped_array = np.array(ecg_actuals_reshaped)
        input_reshaped = ppg.squeeze()
        input_reshaped_array = np.array(input_reshaped)

        random_index = np.random.randint(0, len(ecg_predictions_reshaped_array))
        ppg_scaling_factor = config['parameters']['ppg_scaling_factor']

        # Select the corresponding actual and predicted ECG signals
        ecg_predictions_random = ecg_predictions_reshaped_array[random_index]  # Predicted ECG signal
        ecg_actuals_random = ecg_actuals_reshaped_array[random_index]  # Actual ECG signal
        input_random = input_reshaped_array[random_index]
    
        fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

        sampling_frequency = 256
        time = np.arange(0, sequence_length) / sampling_frequency

        print(input_random)
        print(input_random.shape)
        print(type(input_random))


        # Plot for Green PPG signals while sitting
        axes[0].plot(
            time,
            ecg_actuals_random,
            label='Actual ECG',
            #color='g'
        )
        axes[0].plot(
            time,
            ecg_predictions_random,
            label='Predicted ECG',
            #color='g'
        )
        axes[0].set_xlabel("Time in seconds")
        axes[0].set_ylabel("Normalized amplitude")
        axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

        # Plot for Green PPG signals while running
        axes[1].plot(
            time,
            input_random,
            label='PPG',
            color='g'
        )
        axes[1].set_xlabel("Time in seconds")
        axes[1].set_ylabel("Normalized amplitude")
        axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        plt.savefig(f"../img/wrist_predictions/{model_name}_ecg.pdf")


        print("Evaluation finished!")

    # Wrist and IMU
    elif (config['general']['train'] == False) and (loaded_config['general']['dataset']=='Wrist') and (loaded_config['general']['imu']==True):
        ecg_predictions_reshaped = ecg_predictions.squeeze()
        ecg_predictions_reshaped_array = np.array(ecg_predictions_reshaped)
        ecg_actuals_reshaped = ecg_actuals.squeeze()
        ecg_actuals_reshaped_array = np.array(ecg_actuals_reshaped)
        input_reshaped = ppg.squeeze()
        input_reshaped_array = np.array(input_reshaped)

        random_index = np.random.randint(0, len(ecg_predictions_reshaped_array))
        ppg_scaling_factor = config['parameters']['ppg_scaling_factor']

        # Select the corresponding actual and predicted ECG signals
        ecg_predictions_random = ecg_predictions_reshaped_array[random_index]  # Predicted ECG signal
        ecg_actuals_random = ecg_actuals_reshaped_array[random_index]  # Actual ECG signal
        input_random = input_reshaped_array[random_index]
    
        fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

        sampling_frequency = 256
        time = np.arange(0, sequence_length) / sampling_frequency


        ppg_single = input_random[:, 0]   # R PPG
        a_x_low = input_random[:, 1]
        a_y_low = input_random[:, 2]
        a_z_low = input_random[:, 3]
        a_x_wide = input_random[:,4]
        a_y_wide = input_random[:,5]
        a_z_wide = input_random[:, 6]
        g_x = input_random[:, 7]
        g_y = input_random[:, 8]
        g_z = input_random[:, 9]


        # Plot for Green PPG signals while sitting
        axes[0].plot(
            time,
            ecg_actuals_random,
            label='Actual ECG',
            #color='g'
        )
        axes[0].plot(
            time,
            ecg_predictions_random,
            label='Predicted ECG',
            #color='g'
        )
        axes[0].set_xlabel("Time in seconds")
        axes[0].set_ylabel("Normalized amplitude")
        axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

        # Plot for Green PPG signals while running
        axes[1].plot(
            time,
            ppg_single,
            label='PPG',
            color='g'
        )
        axes[1].set_xlabel("Time in seconds")
        axes[1].set_ylabel("Normalized amplitude")
        axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        plt.savefig(f"../img/wrist_predictions/{model_name}_ecg.pdf")


        # fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns

        # # Plot for Green PPG signals while sitting
        # axes[0].plot(
        #     time,
        #     ecg_actuals_random,
        #     label='Actual ECG',
        #     #color='g'
        # )
        # axes[0].plot(
        #     time,
        #     ecg_predictions_random,
        #     label='Predicted ECG',
        #     #color='g'
        # )
        # axes[0].set_xlabel("Time in seconds")
        # axes[0].set_ylabel("Normalized amplitude")
        # axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

        # # Plot for Green PPG signals while running
        # axes[1].plot(
        #     time,
        #     ppg,
        #     label='PPG',
        #     color='g'
        # )
        # axes[1].set_xlabel("Time in seconds")
        # axes[1].set_ylabel("Normalized amplitude")
        # axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        # plt.savefig(f"../img/wrist_predictions/{model_name}_ecg.pdf")

        fig, axes = plt.subplots(1, 2)  # 1 row, 2 columns
        # Plot for Green PPG signals while sitting
        axes[0].plot(
            time,
            a_x_low,
            label='a_x_low',
            color='#4b0082'
        )
        axes[0].plot(
            time,
            a_y_low,
            label='a_y_low',
            color='#8b4513'
        )
        axes[0].plot(
            time,
            a_z_low,
            label='a_z_low',
            color='#808000'
        )
        # axes[0].plot(
        #     time,
        #     a_x_wide,
        #     label='a_x_wide',
        #     color='#DC143C'
        # )
        # axes[0].plot(
        #     time,
        #     a_y_wide,
        #     label='a_y_wide',
        #     color='#D2691E'
        # )
        # axes[0].plot(
        #     time,
        #     a_z_wide,
        #     label='a_z_wide',
        #     color='#DAA520'
        # )
        axes[0].set_xlabel("Time in seconds")
        axes[0].set_ylabel("Normalized amplitude")
        axes[0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=6)

        # Plot for Green PPG signals while running
        axes[1].plot(
            time,
            g_x,
            label='g_x',
            color='c'
        )
        axes[1].plot(
            time,
            g_y,
            label='g_y',
            color='m'
        )
        axes[1].plot(
            time,
            g_z,
            label='g_z',
            color='y'
        )
        axes[1].set_xlabel("Time in seconds")
        axes[1].set_ylabel("Normalized amplitude")
        axes[1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        plt.savefig(f"../img/wrist_predictions/{model_name}_imu.pdf")

        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 14))  # 2 rows, 2 columns

        # Plot ECG signals on the top left
        axes[0, 0].plot(time, ecg_actuals_random, label='Actual ECG')
        axes[0, 0].plot(time, ecg_predictions_random, label='Predicted ECG')
        axes[0, 0].set_xlabel("Time in seconds")
        axes[0, 0].set_ylabel("Normalized amplitude")
        axes[0, 0].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=2)

        # Plot PPG signals on the top right
        axes[0, 1].plot(time, ppg_single, label='PPG', color='g')
        axes[0, 1].set_xlabel("Time in seconds")
        axes[0, 1].set_ylabel("Normalized amplitude")
        axes[0, 1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        # Plot accelerometer signals on the bottom left
        axes[1, 0].plot(time, a_x_low, label='a_x_low', color='#4b0082')
        axes[1, 0].plot(time, a_y_low, label='a_y_low', color='#8b4513')
        axes[1, 0].plot(time, a_z_low, label='a_z_low', color='#808000')        
        axes[1, 0].plot(time, a_x_wide,label='a_x_wide', color='#DC143C', linestyle='dashed')
        axes[1, 0].plot(time, a_y_wide,label='a_y_wide', color='#D2691E', linestyle='dashed')
        axes[1, 0].plot(time, a_z_wide,  label='a_z_wide',color='#DAA520', linestyle='dashed')
        axes[1, 0].set_xlabel("Time in seconds")
        axes[1, 0].set_ylabel("Normalized amplitude")
        axes[1, 0].legend(bbox_to_anchor=(1, 1.20), loc='upper right', frameon=False, ncol=3)

        # Plot gyroscope signals on the bottom right
        axes[1, 1].plot(time, g_x, label='g_x', color='c')
        axes[1, 1].plot(time, g_y, label='g_y', color='m')
        axes[1, 1].plot(time, g_z, label='g_z', color='y')
        axes[1, 1].set_xlabel("Time in seconds")
        axes[1, 1].set_ylabel("Normalized amplitude")
        axes[1, 1].legend(bbox_to_anchor=(1, 1.10), loc='upper right', frameon=False, ncol=3)

        # Adjust layout and save the figure
        plt.tight_layout()

        plt.savefig(f"../img/wrist_predictions/{model_name}_ecg_ppg_acc_gyro.pdf")
        print("Evaluation finished!")

    else:
        print("Invalid training input paramter for evaluation!")