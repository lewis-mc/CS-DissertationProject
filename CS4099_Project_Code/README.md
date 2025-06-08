# README

## Setup:
- Setup a python virtual environment by executing 'python -m venv my_env'
- Activate virutal environment by executing '. my_env/bin/activate'
- Navigate to 'src' directory
- Install all requirements of the project by executing 'pip install -r requirements.txt'
- Use config.py file to alter configuration settings of 'run_mode', 'updated_basleine' and 'hard_negative_mining'
- Baseline Model: updated_basleine = False and hard_negative_mining = False
- Updated Baseline Model: updated_basleine = True and hard_negative_mining = False
- Baseline Model (with Hard Negative Mining): updated_basleine = False and hard_negative_mining = True
- Updated Baseline Model (with Hard Negative Mining): updated_basleine = True and hard_negative_mining = True
- run_mode = Test, also performs evaluation on the test dataset once training has converged
- Due to storage size the dataset has not been submitted with the solution
- To replicate, ensure the dataset can be used create the split_data directory '../../split_data' with sub directories: Train,Evaluate,Test
- Further use data_augment.py to create augmentation images in the '../../data_augment' and ../../data_augment_updated' directory

## How to run:
- Ensure in the 'src' directory
- Execute 'python main.py' to perform training and evaluation
- Results will be outputted to a the experiment_name directory (defined in config.py)