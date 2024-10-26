import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd


def read_img_as_csv(datasetname = "handwritten-digits"):
    # Path to the folder where the dataset will be saved   
    # datasetname = "handwritten-digits" 
    data_path = os.path.join(os.getcwd(), f'data/{datasetname}')

    features = []
    labels = []
    cleanedFile = []
    for dir in tqdm(os.listdir(data_path)):
        imgDir = os.path.join(data_path, dir)
        img_label = int(str(imgDir).split('_')[1]) # digit_8  => | digit | 8 |
        for img_n in os.listdir(imgDir):
            img_path = os.path.join(imgDir, img_n)
            img =cv2.imread(img_path)
            if img is None:
                cleanedFile.append(img_path)
            else:
                img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)    #Convert color image into gray scale to detect edges better
                img = cv2.resize(img , (100,100))
                features.append(img)            
                labels.append(img_label)


    print(f"Total removed file due to read function: {len(cleanedFile)}")



    features = np.array(features)
    labels = np.array(labels)
    features = features/features.max()

    print(f"X matrix shape: {features.shape}")
    print(f"Y matrix shape: {labels.shape}")

    random = np.random.randint(len(labels)) 
    print(random)
    plt.imshow(features[random], cmap='gray')
    plt.title(labels[random])
    plt.show()
    datadf = pd.DataFrame(features.reshape(features.shape[0], features.shape[1] * features.shape[2]))
    datadf.insert(0,"Labels" ,labels)
    print(datadf.head())
    savedPath = os.path.join(os.getcwd(), f'{datasetname}')
    datadf.to_csv(f'{savedPath}\{datasetname}.csv')
    print(f'data has been stored in: {savedPath}\{datasetname}.csv')
    return datadf


if __name__ == '__main__':
         read_img_as_csv()












