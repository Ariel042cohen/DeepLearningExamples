import os
import numpy as np
from imageio import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.optimizers import Adam
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

cats_train = []

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="cats_and_dogs_filtered/train",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="cats_and_dogs_filtered/validation",target_size=(224,224))
'''
for file in os.listdir('E:/PrimoseDeep/cats_and_dogs_filtered/train/cats'):
    cat = imread("E:/PrimoseDeep/cats_and_dogs_filtered/train/cats/" + file)
    cat = resize(cat, (224, 224))
    cats_train.append(cat)

cats_train_vector = np.stack(cats_train, axis=0 )

dogs_train = []

for file in os.listdir('E:/PrimoseDeep/cats_and_dogs_filtered/train/dogs'):
    dog = imread("E:/PrimoseDeep/cats_and_dogs_filtered/train/dogs/" + file)
    dog = resize(dog, (224, 224))
    dogs_train.append(dog)

dogs_train_vector = np.stack(dogs_train, axis=0)

cats_test = []

for file in os.listdir('E:/PrimoseDeep/cats_and_dogs_filtered/validation/cats'):
    cat = imread("E:/PrimoseDeep/cats_and_dogs_filtered/validation/cats/" + file)
    cat = resize(cat, (224, 224))
    cats_test.append(cat)

cats_test_vector = np.stack(cats_test, axis=0 )

dogs_test = []

for file in os.listdir('E:/PrimoseDeep/cats_and_dogs_filtered/validation/dogs'):
    dog = imread("E:/PrimoseDeep/cats_and_dogs_filtered/validation/dogs/" + file)
    dog = resize(dog, (224, 224))
    dogs_test.append(dog)

dogs_test_vector = np.stack(dogs_test, axis=0)

X_train = np.concatenate((cats_train_vector, dogs_train_vector), axis=0)
y_train = np.concatenate((np.ones((cats_train_vector.shape[0], 1)), np.zeros((dogs_train_vector.shape[0], 1))), axis=0)

X_test = np.concatenate((cats_test_vector, dogs_test_vector), axis=0)
y_test = np.concatenate((np.ones((cats_test_vector.shape[0], 1)), np.zeros((dogs_test_vector.shape[0], 1))), axis=0)

#pickle.dump(X_train, open("X_train.pickle","wb"))
#pickle.dump(y_train, open("y_train.pickle","wb"))
#pickle.dump(X_test, open("X_test.pickle","wb"))
#pickle.dump(y_test, open("y_test.pickle","wb"))

#pickle_out.close()
'''

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])

pyplot.plot(hist.history["acc"])
pyplot.plot(hist.history['val_acc'])
pyplot.plot(hist.history['loss'])
pyplot.plot(hist.history['val_loss'])
pyplot.title("model accuracy")
pyplot.ylabel("Accuracy")
pyplot.xlabel("Epoch")
pyplot.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
pyplot.show()

# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))