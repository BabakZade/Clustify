
import os
from datasets import load_dataset


# Set a custom cache path
datasetname = 'text_emotion_recognition'
download_path = os.path.join(os.getcwd(), f'data/{datasetname}')
emotions = load_dataset("SetFit/emotion", cache_dir=download_path)

# Save each split as a CSV file
emotions['train'].to_csv(os.path.join(download_path, 'train.csv'))
emotions['validation'].to_csv(os.path.join(download_path, 'validation.csv'))
emotions['test'].to_csv(os.path.join(download_path, 'test.csv'))


