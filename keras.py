# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:16:59 2021

@author: ariel
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

diabetes = pd.read_csv('diabetes.csv')

print(diabetes.head())

model = Sequential(
    [
        Dense(8, input_dim=8, activation="relu", name="hidden_layer"),
        Dense(1, name="layer3", activation="sigmoid"),
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Call model on a test input
X = diabetes.drop(columns=['Outcome']).to_numpy()
X = StandardScaler().fit_transform(X)
y = diabetes.loc[:, "Outcome"].to_numpy().T.reshape(768, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test), verbose=0)

pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
