
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras import Sequential


IMAGE_SIZE=256
BATCH_SIZE=32

dataset=tf.keras.preprocessing.image_dataset_from_directory("./PlantVillage",image_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=BATCH_SIZE,shuffle=True)

class_name = dataset.class_names
print(class_name)

len(dataset)


train_size=int(len(dataset)*0.8)
train_ds=dataset.take(train_size)
len(train_ds)

val_size=int(0.1*len(dataset))
val_ds=dataset.skip(train_size).take(val_size)
len(val_ds)

test_ds=dataset.skip(train_size+val_size)
len(test_ds)

model=Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(256,256,3)),
    MaxPooling2D((2,2)),

    Conv2D(64,(3,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3)),
    MaxPooling2D((2,2)),
    Dropout(0.5),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(3,activation='softmax'),


])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

history=model.fit(train_ds,epochs=20,verbose=1,batch_size=BATCH_SIZE,
                  validation_data=val_ds)

loss, accuracy = model.evaluate(test_ds)
print(f"\nTest accuracy: {round(accuracy * 100, 2)}%")

model.save("modelsaved.h5")



