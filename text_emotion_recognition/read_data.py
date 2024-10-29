import os

import pandas as pd

# Function to read images, preprocess, and save as CSV
def read_img_as_csv(datasetname="text_emotion_recognition", filename = ['train.scv', 'test.scv', 'validation.csv']):
    # Path to the folder where the dataset images are stored
    data_path = os.path.join(os.getcwd(), f'data/{datasetname}')

    df_train = pd.read_csv(f'{data_path}//{filename[0]}')
    df_test = pd.read_csv(f'{data_path}//{filename[1]}')
    df_validation = pd.read_csv(f'{data_path}//{filename[2]}')



    return df_train, df_test, df_validation  # Return the DataFrame

# Execute function if script is run directly
if __name__ == '__main__':
    read_img_as_csv()
