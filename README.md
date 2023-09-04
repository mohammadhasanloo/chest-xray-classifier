# NN-CA2-Shallow-CNN-and-Classification-of-Chest-x-ray-Images
### 1.Shallow Convolutional Neural Network for Image Classification [Link](#Part-1-Shallow-Convolutional-Neural-Network-for-Image-Classification)

### 2.Chest X-Ray Image Classification for Pneumonia Detection [Link](#Part-2-Chest-X-Ray-Image-Classification-for-Pneumonia-Detection)

# Part 1: Shallow Convolutional Neural Network for Image Classification

In this section, we will implement a shallow convolutional neural network (SCNNB) for image classification based on the research article titled "Shallow Convolutional Neural Network for Image Classification" available at [Springer](https://link.springer.com/article/10.1007/s42452-019-1903-4).

## Data Preprocessing

In the article, it is mentioned that images from MNIST, FashionMNIST, and 10CIFAR datasets are used. These images are randomly flipped with a 0.5 probability and used as training and test data. According to the article, the size of the images for MNIST and FashionMNIST datasets is 28x28, and for the 10CIFAR dataset, it's 32x32.

Additionally, these datasets are normalized using max-min normalization with a mean and standard deviation both set to 0.5. This normalization helps bring pixel values into a similar range, making it easier for the model to learn features and preventing issues like gradient vanishing or exploding. To apply this normalization, you can add the following transform to your code: `((0.5,), (0.5,))Normalize.transform`.

## Model Architecture

The SCNNB architecture consists of two convolutional layers, two max-pooling layers with a size of 2x2, one fully connected layer, and one softmax layer. Batch Normalization is added after each convolutional layer to improve network training and generalization.

To keep the model shallow, 3x3 kernels are used for both convolutional layers. The first layer has 32 filters, and the second layer has 64 filters. After each Batch Normalization layer, ReLU activation is applied to prevent the model from becoming too linear.

Finally, a fully connected layer with 1280 neurons and a dropout probability of 0.5 is used to prevent overfitting. The output layer employs softmax activation for multi-class classification.

## Hyperparameters

The hyperparameters mentioned in the article are as follows:

- Learning rate: 0.02
- Optimizer: Stochastic Gradient Descent (SGD) with momentum (0.9) and weight decay (0.000005)
- Dropout rate: 0.5
- Number of epochs: 150 for MNIST, 300 for FashionMNIST, and 300 for 10CIFAR
- Batch size: 128
- Loss function: Cross-Entropy Loss

## Model Architectures

The article mentions three proposed architectures: SCNNB, a-SCNNB (with Batch Normalization only after the first convolutional layer), and b-SCNNB (with Batch Normalization removed from both convolutional layers).

## Results

Results show that for MNIST, the model achieves an accuracy of approximately 99.1%, for FashionMNIST, approximately 92.3%, and for 10CIFAR, approximately 78%. Overfitting is observed as the training accuracy reaches 100% after some epochs, but the validation accuracy plateaus. More training data may help further improve performance.

## Conclusion

In conclusion, the SCNNB architecture demonstrates strong performance in image classification tasks, with the potential for further optimization and adaptation to different datasets.

# Part 2: Chest X-Ray Image Classification for Pneumonia Detection

This repository contains code for classifying chest X-ray images to detect pneumonia. The code is based on the following research article:

**Title**: [Automated Diagnosis of Pneumonia from Classification of Chest X-Ray Images using EfficientNet](https://www.researchgate.net/profile/Nusrat-Jahan-122/publication/351643298_Automated_Diagnosis_of_Pneumonia_from_Classification_of_Chest_X-Ray_Images_using_EfficientNet/links/60bf9e35a6fdcc512815ddae/Automated-Diagnosis-of-Pneumonia-from-Classification-of-Chest-X-Ray-Images-using-EfficientNet.pdf)

## Preparing and Preprocessing the Data

### Data Preparation

The dataset used in this article consists of 5863 images, comprising 4273 images from individuals with pneumonia and 1583 images from healthy individuals. Given the complexity of the problem and the architecture of the network, the data is split into three parts: training, validation, and testing, ensuring that the class ratio is maintained in all sections to prevent imbalance. The data is divided into three parts with proportions of 60%, 20%, and 20%, respectively. The images are resized to 128x128 pixels.

To augment the dataset and address data scarcity, data augmentation techniques such as zooming, rotation, horizontal or vertical shifting, and more are applied. Additionally, all data is rescaled by a factor of 1/255, which is equivalent to min-max normalization based on the pixel value range.

### Data Retrieval

The "json.Kaggle" file is downloaded from Kaggle to be used with the Kaggle API and uploaded to the Google Colab environment. The dataset is then downloaded in the Google Colab environment.

### Data Loading and Preprocessing

Images are loaded using OpenCV, resized to the specified dimensions (128x128) as mentioned in the article, and stored in an array. Data is divided by 255 according to the rescale parameter. The dataset is then split into three parts using the StratifiedShuffleSplit function from the sklearn library, ensuring that the class ratio is maintained in all sections.

### Data Augmentation

Data augmentation is implemented using the ImageDataGenerator in the Keras library.

## Model Architecture

A convolutional neural network (CNN) architecture called EfficientNet is used in this project. The goal of EfficientNet is to achieve better performance in image classification with fewer parameters compared to previous architectures. The main idea behind EfficientNet is to scale the depth, width, and resolution of the network in a principled way to balance model performance and computational cost. Specifically, it employs the Compound Scaling method to simultaneously scale the depth, width, and resolution based on predefined coefficients.

The overall architecture of the network consists of three main parts:

1. **Stem Layer**: This section includes a convolutional layer followed by a batch normalization layer and a ReLU activation function. The goal of this layer is to extract low-level features from input images.

2. **Repeated Blocks**: These blocks are the main building blocks of the EfficientNet architecture. Each block consists of a series of layers, including convolutional layers, batch normalization, and activation functions. The depth, width, and resolution of each block are scaled according to the predefined coefficients. The number of these blocks is determined by a parameter called the coefficient depth.

3. **Classification Head**: This section is the final layer of the EfficientNet architecture, consisting of a Global Average Pooling layer followed by a fully connected layer with a softmax activation function. The purpose of this layer is to map the extracted feature vectors from the repeated blocks to the class output and determine the corresponding class.

EfficientNet is designed with a principled approach that allows it to achieve better accuracy with fewer computational resources and lower complexity compared to older CNN architectures. It has been widely tested on large image classification benchmarks such as ImageNet and consistently demonstrated good performance. Furthermore, the idea of balancing model capability and computational cost makes it a suitable choice for researchers looking to achieve good accuracy with lower computational expenses.

### Implementation

The code for the EfficientNet model is implemented using the Keras library with the functional API. The EfficientNetB2 model is loaded with its pre-trained weights on the ImageNet dataset, excluding the top classification layer. The input tensor dimensions are set to 128x128, as mentioned in the article. The final layers of the network are then connected to the end of EfficientNetB2.

### Training Details

In the results section of the article, it is mentioned that the Adam optimizer with a learning rate of 0.001 achieved good results. The batch size is set to 128. Class weights are calculated to address the class imbalance issue in the dataset. It is also mentioned in the article that some of the top layers (at the end) of the EfficientNetB2 model were allowed to be fine-tuned, but the lower layers were frozen. The exact number of these layers is not specified. In our implementation, we allowed the training of the last 80 layers, considering that EfficientNetB2 has around 340 layers in total, and allowing at least 25% of the layers to be trainable is a reasonable choice to update higher-level filters. EarlyStopping and ModelCheckpoint were used in the code, but EarlyStopping was disabled due to model instability in the early training steps. ModelCheckpoint was also not used due to frequent interruptions in the training process.

The model is trained with the following hyperparameter changes, which led to an improved model with reasonable accuracy:

1. Increased the number of layers allowed for training, leaving only the first 74 layers frozen, as the stem layers (responsible for extracting low-level features) and the first two blocks of repeated blocks are allowed to be trained.

2. Reduced the learning rate significantly, as increasing the number of trainable layers and parameters can lead to rapid instability or overfitting. A learning rate scheduler was used to further reduce the learning rate in two stages.

3. Reduced the batch size to 32 to achieve faster convergence.

The model is trained for 20 epochs.

## Results

The accuracy of the model on the test data is approximately 82%. While this may not seem high at first glance, it is important to consider the challenges of the task and the limited amount of training data available for fine-tuning a large architecture like EfficientNet. Further improvements could be explored with larger datasets or more extensive fine-tuning.
