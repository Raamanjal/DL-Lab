import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

# Normalize

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


# One Hot Encoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Data Augmentation


datagen = ImageDataGenerator(

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1

)

datagen.fit(X_train)




# CNN Model


model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), strides=(1,1),padding='same',),

    MaxPooling2D(2,2),

    Dropout(0.25),

    Conv2D(64, (3,3),strides=(1,1),padding='same', activation='relu'),

    MaxPooling2D(2,2),

    Dropout(0.25),

    Flatten(),
    
    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])



# Compile


model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)


# Early Stopping


early_stop = EarlyStopping(

    monitor='val_loss',

    patience=3

)



# Train


history = model.fit(

    datagen.flow(X_train,y_train,batch_size=64),

    epochs=15,

    validation_data=(X_test,y_test),

    callbacks=[early_stop]

)



# Evaluate


loss, acc = model.evaluate(X_test,y_test)

print("\nFinal Accuracy:", acc)



# Plot Loss


plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")

plt.legend(["Train","Validation"])

plt.show()


# PLOT ACCURACY

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Accuracy Curve")

plt.legend(["Train","Validation"])

plt.show()