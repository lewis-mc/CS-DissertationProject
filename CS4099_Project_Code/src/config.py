# Default configuration for the project

# Alter as needed
experiment_name = 'experiment_1'
run_mode = "train" # "train" or "test"
updated_baseline = True # or False
hard_negative_mining = True # or False

# FINAL
train_dir = '../../split_data/Train'
eval_dir = '../../split_data/Evaluate'
test_dir = '../../split_data/Test'
augment_dir = '../../data_augment/Train'
augment_dir_updated = '../../data_augment_updated'
output_path = f'../../{experiment_name}_output'
output_json_file = f'{output_path}/{experiment_name}.json'
output_json_file_extra = f'{output_path}/{experiment_name}_extra.json'
log_filename = f'{output_path}/{experiment_name}.log'
saved_model_path = f'{output_path}/{experiment_name}_saved_model.pth'
num_classes = 21
num_epochs = 100
batch_size = 64 
learning_rate = 0.00001
num_workers = 4
weight_decay = 0.0001 # Updated Baseline
dropout = 0.25 # Updated Baseline
confidence_threshold = 0.65 # Used in Hard Negative Mining
augment_prob = 0.5
augment_prob_updated = 0.7 # Updated Baseline
patience = 3
