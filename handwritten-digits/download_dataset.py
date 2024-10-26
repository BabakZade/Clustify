import os
import zipfile

# Path to the folder where the dataset will be saved
datasetname = "handwritten-digits"
download_path = os.path.join(os.getcwd(), f'data/{datasetname}')

# Ensure the 'data' folder exists
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Kaggle dataset API command (replace 'dataset-owner/dataset-name' with the actual dataset)
os.system(f'kaggle datasets download -d artgor/{datasetname} -p {download_path}')

# Unzipping the dataset if it's zipped
zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
for zip_file in zip_files:
    zip_file_path = os.path.join(download_path, zip_file)
    
    # Unzipping with Python's zipfile module
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    
    # Remove the zip file after extraction
    os.remove(zip_file_path)
