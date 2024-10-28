
import sklearn.model_selection
from readFile import *
from data_wrangling import *
from pathlib import Path
import pandas as pd
import keras
import sklearn 
import matplotlib.pyplot as plt




datasetname = "handwritten-digits" 
cwd = os.path.join(os.getcwd(), f'{datasetname}')
pathtofile = f'{cwd}\\{datasetname}.csv'
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


def get_model(path_to_model,x_train, y_train, x_test, y_test):
    if Path(path_to_model).is_file():
        model = keras.models.load_model(path_to_model)

    else:
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

        model.add(keras.layers.Dense(labels.shape[1] , activation='softmax'))

        model.compile(optimizer = 'sgd' , 
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])       


        model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))
        model.save(path_to_model)

    return model

path_to_model = f'{cwd}\\model.keras'

model = get_model(path_to_model=path_to_model, x_train=x_train, y_train= y_train, x_test= x_test, y_test=y_test)

loss_train,accuracy_train = model.evaluate(x_train,y_train)
loss_test,accuracy_test = model.evaluate(x_test,y_test)


x_rnd = np.random.randint(y_test.shape[0]) 
y_pr = model.predict(x_test[x_rnd].reshape(1,-1))
y_real=np.argmax(y_test[x_rnd,:])
plt.imshow(x_test[x_rnd].reshape(int(np.sqrt(len(x_test[x_rnd]))), -1), cmap='gray')
plt.title(f'y_pr = {np.argmax(y_pr)}, y = {y_real}')
plt.show()
