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

-Load image data from the "PlantVillage" directory using image_dataset_from_directory.

-Split the dataset into training, validation, and test sets.

-Perform data augmentation on the training set using random flips and rotations.

2.Model Construction:

Build a CNN model using the Sequential API with convolutional layers, max-pooling layers, flattening layer, and dense layers.

Use ReLU activation functions for convolutional layers and softmax activation for the output layer.

Compile the model with the Adam optimizer and Sparse Categorical Crossentropy loss.

3.Model Training:

Train the model using the training dataset and validate it on the validation dataset.

Monitor and record accuracy and loss metrics during training.

4.Model Evaluation:

Evaluate the trained model on the test dataset using the evaluate method.

5.Visualization:

Plot training and validation accuracy and loss over epochs using Matplotlib.

6.Transfer Learning with VGG16:

Implement transfer learning using the VGG16 pre-trained model from Keras applications.

Freeze the layers of the pre-trained model and add new dense layers for classification.

Compile and train the new model on the dataset.

7.Transfer Learning Model Evaluation:

Evaluate the transfer learning model on the test dataset and print the evaluation results.

8.Prediction:

Make predictions on a sample batch from the test dataset and visualize the results.

