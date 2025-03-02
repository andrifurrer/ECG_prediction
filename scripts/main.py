import os
import yaml
from scripts.encoder_decoder_train import train
from scripts.encoder_decoder_evaluation import eval
from encoder_train import encoder_train
from encoder_evaluation import encoder_eval

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Ensure the results, checkpoint, model_summary and model folders exists
    results_folder = config['output']['results'] + config['output']['model_family'] + config['general']['model_name']
    os.makedirs(results_folder, exist_ok=True)
    checkpoints_folder = config['output']['checkpoints'] + config['output']['model_family'] + config['general']['model_name']
    os.makedirs(checkpoints_folder, exist_ok=True)
    model_summary_folder = config['output']['model_summary'] + config['output']['model_family'] + config['general']['model_name']
    os.makedirs(model_summary_folder, exist_ok=True)
    model_folder  = f"../models/{config['output']['model_family']}"
    os.makedirs(model_folder, exist_ok=True)

    if config['general']['model_type'] == 'encoder':
        # Train the model
        if config['general']['train'] == True:
            encoder_train(config)

        # Evaluate the model
        if config['general']['eval'] == True:
            encoder_eval(config) 

    elif config['general']['model_type'] == 'encoder_decoder':
        # Train the model
        if config['general']['train'] == True:
            train(config)

        # Evaluate the model
        if config['general']['eval'] == True:
            eval(config) 
    else:
        print("Model type not supported!")