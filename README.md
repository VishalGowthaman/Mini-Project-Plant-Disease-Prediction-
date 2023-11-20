# MINI PROJECT-PLANT DISEASE RECOGNITION USING CNN
## Aim:
This project aims to develop a CNN model that can accurately identify and classify different diseases that affect crops.
## Features of CNN:
1.Convolutional Layers:
  
  CNNs use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data. These layers apply filters to input data, enabling the detection of patterns and features.

2.Pooling Layers:
  
  Pooling layers reduce the spatial dimensions of the input data by down-sampling, retaining the most important information. Max pooling, for example, selects the maximum value from a group of neighboring pixels, further emphasizing key features.

3.Fully Connected Layers:
  
  Fully connected layers process the high-level features extracted by the convolutional and pooling layers, enabling the model to make predictions or classifications based on the learned representations.

4.Activation Functions:
  
  Non-linear activation functions, such as ReLU (Rectified Linear Unit), introduce non-linearity to the model, allowing it to learn complex relationships in the data. ReLU is commonly used in CNNs for its simplicity and effectiveness.
## Requirements:
1.Image Dataset.

2.Python 3.x for project development.
## Flow Chart:
![image](https://github.com/VishalGowthaman/Mini-Project-Plant-Disease-Prediction-/assets/94165380/3d7e9885-d47f-4da0-8291-c5070bd910ce)
## Algorithm:
1.Data Loading and Preprocessing:

  - Load image data from the "PlantVillage" directory using image_dataset_from_directory.

  - Split the dataset into training, validation, and test sets.

  - Perform data augmentation on the training set using random flips and rotations.

2.Model Construction:

Build a CNN model using the Sequential API with convolutional layers, max-pooling layers, flattening layer, and dense layers.

  - Use ReLU activation functions for convolutional layers and softmax activation for the output layer.

  - Compile the model with the Adam optimizer and Sparse Categorical Crossentropy loss.

3.Model Training:

  - Train the model using the training dataset and validate it on the validation dataset.

  - Monitor and record accuracy and loss metrics during training.

4.Model Evaluation:

  - Evaluate the trained model on the test dataset using the evaluate method.

5.Visualization:

  - Plot training and validation accuracy and loss over epochs using Matplotlib.

6.Transfer Learning with VGG16:

  - Implement transfer learning using the VGG16 pre-trained model from Keras applications.

  - Freeze the layers of the pre-trained model and add new dense layers for classification.

  - Compile and train the new model on the dataset.

7.Transfer Learning Model Evaluation:

  - Evaluate the transfer learning model on the test dataset and print the evaluation results.

8.Prediction:

  - Make predictions on a sample batch from the test dataset and visualize the results.
## Program:
```
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=20

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
class_names

for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")

len(dataset)

train_size = 0.8
len(dataset)*train_size

train_ds = dataset.take(54)
len(train_ds)

test_ds = dataset.skip(54)
len(test_ds)

val_size=0.1
len(dataset)*val_size

val_ds = test_ds.take(6)
len(val_ds)

test_ds = test_ds.skip(6)
len(test_ds)

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

len(train_ds)

len(val_ds)

len(test_ds)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])
model.build(input_shape=input_shape)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=20,
)

scores = model.evaluate(test_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model

base_model = VGG16(input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
                   include_top = False,
                   weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(3, activation = "softmax")(x)

model1 = Model(base_model.input,x)

model1.summary()

model1.compile(optimizer = "adam", 
              loss = "SparseCategoricalCrossentropy",
              metrics = ["accuracy"])

history = model1.fit(x = train_ds,
                 epochs = 15)

print(model1.evaluate(x = test_ds))

import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model1.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
```

