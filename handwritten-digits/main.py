import sklearn.model_selection
from readFile import *          # Custom function to read images
from data_wrangling import *    # Custom function for data wrangling
from pathlib import Path
import pandas as pd
import keras
import sklearn 
import matplotlib.pyplot as plt
import os
import glob

# Define dataset and paths
datasetname = "handwritten-digits" 
cwd = os.path.join(os.getcwd(), f'{datasetname}')  # Set current working directory for dataset
pathtofile = f'{cwd}\\{datasetname}.csv'           # Define full path to CSV file
img_data = pd.DataFrame()                          # Initialize empty DataFrame

# Check if dataset exists as CSV; if not, read and save it
if Path(pathtofile).is_file():
    img_data = pd.read_csv(pathtofile, index_col=0)  # Read CSV if it exists
else:
    img_data = read_img_as_csv(datasetname=datasetname)  # Custom function to create CSV from images

# Data wrangling: preprocess data
img_data = wrangle_data(img_data)

# Print features (excluding labels column)
print(img_data.loc[:, img_data.columns != 'Labels'])

# Separate features and labels
labels = np.array(img_data['Labels'])  # Extract labels as array
features = np.array(img_data.loc[:, img_data.columns != 'Labels'])  # Extract features

# One-hot encode the labels for multi-class classification
labels = keras.utils.to_categorical(labels)  

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.8)

# Model creation or loading function
def get_model(path_to_model, x_train, y_train, x_test, y_test, epoch, batchsize):
    model_name = f'model_{epoch}_{batchsize}.keras'  # Define model name based on epoch and batch size
    
    # Check if model file already exists
    if Path(path_to_model + model_name).is_file():
        model = keras.models.load_model(path_to_model + model_name)  # Load existing model
    else:
        # If model file does not exist, delete any old model files
        keras_files = glob.glob(os.path.join(path_to_model, '*.keras'))
        for file in keras_files:
            os.remove(file)
            print("=============================================================")
            print(f"Deleted: {file}")
            print("=============================================================")

        # Define new Sequential model architecture
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256, activation="relu", input_dim=x_train.shape[1]))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(labels.shape[1], activation='softmax'))  # Output layer with softmax for multi-class

        # Compile the model with optimizer, loss, and metrics
        model.compile(optimizer='sgd', 
                      loss='categorical_crossentropy',
                      metrics=[keras.metrics.Precision(), 
                               keras.metrics.Recall(), 
                               keras.metrics.Accuracy()])       

        # Train the model and save training history
        history = model.fit(x_train, y_train, epochs=epoch, batch_size=batchsize, validation_data=(x_test, y_test))
        model.save(path_to_model + model_name)  # Save the model

    
    return model

# Set training parameters
epoch = 100
batchsize = 16
path_to_model = f'{cwd}\\'

# Get or train the model
model = get_model(path_to_model=path_to_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epoch=epoch, batchsize=batchsize)

# Evaluate model performance on training data
loss_train, accuracy_train = model.evaluate(x_train, y_train)

# Test model on a random sample from the test set
x_rnd = np.random.randint(y_test.shape[0])  # Random index in test set
y_pr = model.predict(x_test[x_rnd].reshape(1, -1))  # Predict class for the random test sample
y_real = np.argmax(y_test[x_rnd, :])  # Get the true class label

# Display the sample as an image with the predicted and actual labels
plt.imshow(x_test[x_rnd].reshape(int(np.sqrt(len(x_test[x_rnd]))), -1), cmap='gray')
plt.title(f'y_pr = {np.argmax(y_pr)}, y = {y_real}')
plt.show()
