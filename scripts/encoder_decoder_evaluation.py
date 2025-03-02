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

from torch.cuda.amp import autocast

# Add the parent directory (i.e. transformer, means parent directory of 'scripts' and 'notebooks') to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

# Import the function
from scripts.utils import *
from scripts.classes import *

def eval(config):
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
    train_subjects = loaded_data["Training subjects"]
    val_subjects = loaded_data["Validation subjects"]
    test_subjects = loaded_data["Test subjects"]
    # X_train = loaded_data["X_train"]  # PyTorch tensor
    # y_train = loaded_data["y_train"]  # PyTorch tensor
    # X_val = loaded_data["X_val"]      # PyTorch tensor
    # y_val = loaded_data["y_val"]      # PyTorch tensor
    X_test = loaded_data["X_test"]      # PyTorch tensor
    y_test = loaded_data["y_test"]      # PyTorch tensor
    # df_train = loaded_data["df_train"]  # Pandas DataFrame
    # df_val = loaded_data["df_val"]      # Pandas DataFrame
    df_test = loaded_data["df_test"] # Pandas DataFrame
    training_loss = loaded_data["Training_loss"]  # Numpy array
    validation_loss = loaded_data["Validation_loss"]  # Numpy array
    epochs = loaded_data["Epochs"]  # Numpy array


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
    model = TransformerTimeSeries(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device) 

    # Load the model
    model.load_state_dict(torch.load(f"../models/{model_family}{model_name}_trained_model.pth", map_location=torch.device(device)))

    ### Validation
    # Initialize storage for aggregated predictions and actual values
    ecg_predictions = []
    ecg_actuals = []
    ppg = []
    subjects = []  # Store subject info for each batch
    actions = []   # Store action info for each batch

    # Loss function: Mean Squared Error for regression tasks
    loss_fn = nn.MSELoss()

    # Test Loss
    test_loss = np.array([])
    running_test_loss = 0

    # Check if the dataloader should be used or not
    if use_dataloader:
        # Convert tensors to Datasets
        test_dataset = PreprocessedDataset(X_test, y_test)
        # Create DataLoaders for each set
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Iterate over the validation set in batches
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        if use_dataloader:
            # Iterate through the batches in the test_loader to load the data in batches
            for batch_idx, (batch_X_test, batch_y_test) in enumerate(test_loader):
                # Move the batch data to the device (GPU or CPU)
                batch_X_test = batch_X_test.to(device)
                batch_y_test = batch_y_test.to(device)
                
                # Get the start and end index of the current batch in df_test
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batch_X_test)
                
                # Retrieve the corresponding (subject, action) pair for this batch from df_test
                batch_subjects = df_test.iloc[start_idx:end_idx]['subject'].values
                batch_actions = df_test.iloc[start_idx:end_idx]['action'].values

                # Initialize decoder input with the start token (all zeros)
                tgt_input_val = torch.zeros((batch_y_test.size(0), 1, batch_y_test.size(-1)), device=device)

                # Autoregressive decoding
                predictions = []
                # Enable mixed precision
                with autocast():
                    for _ in range(batch_y_test.size(1)):
                        # Generate tgt_mask for the current step
                        tgt_mask = generate_square_subsequent_mask(tgt_input_val.size(1)).to(device)

                        # Generate key padding masks
                        src_key_padding_mask = (batch_X_test[:, :, 0] == 0).to(device)
                        tgt_key_padding_mask = (tgt_input_val.squeeze(-1) == 0).to(device)

                        # Forward pass
                        step_output = model(
                            batch_X_test,
                            tgt_input_val,
                            tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                        )[:, -1:, :]  # Take the output for the last timestep
                        predictions.append(step_output)

                        # Append the prediction to tgt_input_val for the next timestep
                        tgt_input_val = torch.cat([tgt_input_val, step_output], dim=1)

                    # Combine all timestep predictions
                    batch_predictions = torch.cat(predictions, dim=1)

                # Create the target input (tgt) for the decoder using teacher forcing
                # tgt_input_val = torch.cat(
                #     [torch.zeros((batch_y_test.size(0), 1, batch_y_test.size(-1)), device=device),  # Start token
                #     batch_y_test[:, :-1, :]],  # Shifted target sequence
                #     dim=1
                # )

                # # Generate tgt_mask 
                # # Shape: [seq_len_tgt, seq_len_tgt]
                # tgt_mask = generate_square_subsequent_mask(tgt_input_val.size(1)).to(device)

                # # Generate src_key_padding_mask 
                # # Shape: [batch_size, seq_len_src]
                # src_key_padding_mask = (batch_X_test[:, :, 0] == 0).to(device) # Use only one feature (i.e. one ppg signal)

                # # Generate tgt_key_padding_mask (pad tokens are 0 in the target)
                # # Shape: [batch_size, seq_len_tgt]
                # tgt_key_padding_mask = (tgt_input_val.squeeze(-1) == 0).to(device)

                # # Debugging: print mask shapes to ensure they are correct
                # # assert src_key_padding_mask.shape == [batch_size, sequence_length]
                # # assert src_key_padding_mask.shape == [batch_size, sequence_length]
                # # print("seq_len_src", sequence_length)
                # # print("tgt_len", sequence_length)
                # # print(f"src_key_padding_mask shape: {src_key_padding_mask.shape}")  # Should be [batch_size, seq_len_src]
                # # print(f"tgt_key_padding_mask shape: {tgt_key_padding_mask.shape}")  # Should be [batch_size, seq_len_tgt]

                # memory_mask = None

                # # Forward pass to get predictions
                # batch_predictions = model(
                #     batch_X_test, 
                #     tgt_input_val, 
                #     tgt_mask=tgt_mask, 
                #     src_key_padding_mask=src_key_padding_mask, 
                #     tgt_key_padding_mask=tgt_key_padding_mask
                # )
                # Calculate loss for this batch
                loss = loss_fn(batch_predictions, batch_y_test)

                # Accumulate total validation loss
                running_test_loss += loss.item() * batch_X_test.size(0)

                # Store predictions, actuals, subjects, and actions
                ecg_predictions.append(batch_predictions.cpu())  # Move to CPU for numpy/scaler operations
                ecg_actuals.append(batch_y_test.cpu())
                ppg.append(batch_X_test.cpu())
                subjects.extend(batch_subjects)
                actions.extend(batch_actions)

        else:
            for j in range(0, len(X_test), batch_size):
                # Get the current validation batch
                batch_X_test = X_test[j:j + batch_size].to(device)
                batch_y_test = y_test[j:j + batch_size].to(device)

                # Retrieve subject and action for the batch
                batch_subjects = df_test.iloc[j:j + batch_size]['subject'].values
                batch_actions = df_test.iloc[j:j + batch_size]['action'].values

                # Create the target input (tgt) for the decoder using teacher forcing
                tgt_input_val = torch.cat(
                    [torch.zeros((batch_y_test.size(0), 1, batch_y_test.size(-1)), device=device),  # Start token
                    batch_y_test[:, :-1, :]],  # Shifted target sequence
                    dim=1
                )

                # Generate tgt_mask 
                # Shape: [seq_len_tgt, seq_len_tgt]
                tgt_mask = generate_square_subsequent_mask(tgt_input_val.size(1)).to(device)

                # Generate src_key_padding_mask 
                # Shape: [batch_size, seq_len_src]
                src_key_padding_mask = (batch_X_test[:, :, 0] == 0).to(device) # Use only one feature (i.e. one ppg signal)

                # Generate tgt_key_padding_mask (pad tokens are 0 in the target)
                # Shape: [batch_size, seq_len_tgt]
                tgt_key_padding_mask = (tgt_input_val.squeeze(-1) == 0).to(device)

                # Debugging: print mask shapes to ensure they are correct
                # assert src_key_padding_mask.shape == [batch_size, sequence_length]
                # assert src_key_padding_mask.shape == [batch_size, sequence_length]
                # print("seq_len_src", sequence_length)
                # print("tgt_len", sequence_length)
                # print(f"src_key_padding_mask shape: {src_key_padding_mask.shape}")  # Should be [batch_size, seq_len_src]
                # print(f"tgt_key_padding_mask shape: {tgt_key_padding_mask.shape}")  # Should be [batch_size, seq_len_tgt]

                memory_mask = None

                # Forward pass to get predictions
                batch_predictions = model(
                    batch_X_test, 
                    tgt_input_val, 
                    tgt_mask=tgt_mask, 
                    src_key_padding_mask=src_key_padding_mask, 
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
                # Calculate loss for this batch
                loss = loss_fn(batch_predictions, batch_y_test)

                # Accumulate total validation loss
                running_test_loss += loss.item() * batch_X_test.size(0)

                # Store predictions, actuals, subjects, and actions
                ecg_predictions.append(batch_predictions.cpu())  # Move to CPU for numpy/scaler operations
                ecg_actuals.append(batch_y_test.cpu())
                ppg.append(batch_X_test.cpu())
                subjects.extend(batch_subjects)
                actions.extend(batch_actions)

    # Average the test loss over all samples
    avg_test_loss = running_test_loss / len(X_test)
    test_rmse = torch.sqrt(torch.tensor(avg_test_loss))
    test_loss = np.append(test_loss, test_rmse.cpu())

    # Concatenate all batches
    ecg_predictions = torch.cat(ecg_predictions, dim=0)
    ecg_actuals = torch.cat(ecg_actuals, dim=0)
    ppg = torch.cat(ppg, dim=0)

    # Initialize lists for original scale data
    ecg_predictions_original_scale = []
    ecg_actuals_original_scale = []
    ppg_original_scale = []

    # Process each sequence
    for i in range(len(ecg_predictions)):
        # Get subject and action for the current sequence
        subject = subjects[i]
        action = actions[i]

        # Retrieve the correct scalers
        scaler_input = scalers[(subject, action)]['input_scaler']
        scaler_target = scalers[(subject, action)]['target_scaler']

        # Inverse transform predictions and actuals for the current sequence
        ecg_pred = ecg_predictions[i].squeeze(-1).numpy()  # Shape: [sequence_length]
        ecg_act = ecg_actuals[i].squeeze(-1).numpy()       # Shape: [sequence_length]
        ppg_seq = ppg[i].numpy()                          # Shape: [sequence_length, 3]

        ecg_predictions_original_scale.append(scaler_target.inverse_transform(ecg_pred.reshape(-1, 1)).flatten())
        ecg_actuals_original_scale.append(scaler_target.inverse_transform(ecg_act.reshape(-1, 1)).flatten())
        ppg_original_scale.append(scaler_input.inverse_transform(ppg_seq))

    # Convert back to arrays
    ecg_predictions_original_scale = np.array(ecg_predictions_original_scale)
    ecg_actuals_original_scale = np.array(ecg_actuals_original_scale)
    ppg_original_scale = np.array(ppg_original_scale)

    # Separate PPG channels 
    red_ppg = ppg_original_scale[:, :, 0]  # Red PPG
    ir_ppg = ppg_original_scale[:, :, 1]   # IR PPG
    green_ppg = ppg_original_scale[:, :, 2]  # Green PPG


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

    random_index = np.random.randint(0, len(ecg_predictions_original_scale))
    ppg_scaling_factor = config['parameters']['ppg_scaling_factor']

    # Select the corresponding actual and predicted ECG signals
    ecg_predictions_random = ecg_predictions_original_scale[random_index]  # Predicted ECG signal
    ecg_actuals_random = ecg_actuals_original_scale[random_index]  # Actual ECG signal

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
    plt.plot(ppg_scaling_factor*red_ppg[random_index], label="Red PPG", alpha=alpha)
    plt.plot(ppg_scaling_factor*ir_ppg[random_index], label="IR PPG", alpha=alpha)
    plt.plot(ppg_scaling_factor*green_ppg[random_index], label="Green PPG", alpha=alpha)
    plt.title(f"ECG Prediction vs Actual (Sequence {random_index}) with PPG signals")
    plt.xlabel('Time Step')
    plt.ylabel('ECG Signal')
    plt.legend()

    plt.savefig(f"{results_folder}/{model_name}_random_seq_ppg.png")

    print("Evaluation finished!")