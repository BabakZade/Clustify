import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

# Function to read images, preprocess, and save as CSV
def read_img_as_csv(datasetname="handwritten-digits"):
    # Path to the folder where the dataset images are stored
    data_path = os.path.join(os.getcwd(), f'data/{datasetname}')

    features = []  # List to store image data
    labels = []    # List to store image labels
    cleanedFile = []  # List to keep track of unreadable files

    # Loop over each directory in the dataset path
    for dir in tqdm(os.listdir(data_path)):
        imgDir = os.path.join(data_path, dir)
        
        # Extract the label from the directory name (e.g., "digit_8" -> label is 8)
        img_label = int(str(imgDir).split('_')[1])  # Assumes directory names follow this format

        # Loop over each image in the directory
        for img_n in os.listdir(imgDir):
            img_path = os.path.join(imgDir, img_n)  # Get the full path of the image

            # Read the image
            img = cv2.imread(img_path)
            if img is None:  # If image is unreadable, add to cleanedFile list
                cleanedFile.append(img_path)
            else:
                # Convert to grayscale for simpler processing
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize the image to a fixed size (100x100)
                img = cv2.resize(img, (100, 100))

                # Append the processed image and its label to the lists
                features.append(img)
                labels.append(img_label)

    # Report the number of unreadable files
    print(f"Total removed files due to read function: {len(cleanedFile)}")

    # Convert features and labels to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Normalize features to scale values between 0 and 1
    features = features / features.max()

    # Print shapes for verification
    print(f"X matrix shape: {features.shape}")
    print(f"Y matrix shape: {labels.shape}")

    # Show a random sample image with its label for verification
    random = np.random.randint(len(labels)) 
    print(random)
    plt.imshow(features[random], cmap='gray')
    plt.title(labels[random])
    plt.show()

    # Convert features array to DataFrame for saving as CSV
    datadf = pd.DataFrame(features.reshape(features.shape[0], features.shape[1] * features.shape[2]))
    datadf.insert(0, "Labels", labels)  # Insert labels as the first column
    print(datadf.head())

    # Define path to save the CSV file
    savedPath = os.path.join(os.getcwd(), f'{datasetname}')
    datadf.to_csv(f'{savedPath}\\{datasetname}.csv')  # Save DataFrame to CSV
    print(f'data has been stored in: {savedPath}\\{datasetname}.csv')

    return datadf  # Return the DataFrame

# Execute function if script is run directly
if __name__ == '__main__':
    read_img_as_csv()
