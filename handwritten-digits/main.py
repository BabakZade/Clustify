
import sklearn.model_selection
from readFile import *
from data_wrangling import *
from pathlib import Path
import pandas as pd
import keras
import sklearn 




datasetname = "handwritten-digits" 
cwd = os.path.join(os.getcwd(), f'{datasetname}')
pathtofile = f'{cwd}/{datasetname}.csv'
img_data = pd.DataFrame()
if Path(pathtofile).is_file():
    img_data = pd.read_csv(pathtofile,index_col=0)

else:
    img_data = read_img_as_csv(datasetname=datasetname)




img_data = wrangle_data(img_data)

print(img_data.loc[:, img_data.columns != 'Labels'])

# model
labels = np.array(img_data['Labels'])
features = np.array(img_data.loc[:, img_data.columns != 'Labels'])




labels = keras.utils.to_categorical(labels) # => turns to matrix

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(features, labels, train_size= 0.8)

model = keras.models.Sequential()
model.add(keras.layers.Dense(256 , activation = "relu" , input_dim=x_train.shape[1]))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(128 , activation = "relu" ))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(128 , activation = "relu" ))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(64 , activation = "relu" ))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(64 , activation = "relu" ))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(10 , activation='softmax'))

model.compile(optimizer = 'sgd' , 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))

