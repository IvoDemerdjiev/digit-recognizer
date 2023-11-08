from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

train = pd.read_csv("../PythonProjects/digit-recognizer/train.csv")
print(train.shape)
#print(train.head())

test = pd.read_csv("../PythonProjects/digit-recognizer/test.csv")
print(test.shape)
#print(test.head())

labels = train.iloc[:,0].values.astype('int32')
X_train = train.iloc[:, 1:].values.astype('float32')

y_train = np_utils.to_categorical(labels)

X_test = test.values.astype('float32')
scale = np.max(X_train)
X_train = X_train / scale
X_test = X_test / scale

mean = np.std(X_train)
X_train = X_train - mean
X_test = X_test - mean

def get_mlp_model():
    model = Sequential([
        Dense(128, input_dim = X_train.shape[1]),
        Activation('relu'),
        Dropout(0.15),
        Dense(128),
        Activation('relu'),
        Dropout(0.15),
        Dense(y_train.shape[1]),
        Activation('softmax')
        ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = get_mlp_model()
print(model)
print("Training...")
model.fit(X_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)


submissions=pd.DataFrame({"ImageId": list(range(1,len(preds)+1)),
                         "Label": preds})
submissions.to_csv("DR.csv", index=False, header=True)
