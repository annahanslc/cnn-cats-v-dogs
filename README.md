# Cats vs Dogs
### Image Classification using Convolutional Neural Network

Having had both dogs and cats in our family for many years, I can confidently tell the difference between the two. Yet, many pet-focused websites still struggle to distinguish between them when users are setting up profiles for their pets.

While booking a stay for our pets before a family vacation, I was asked to upload photos of each pet one by one and manually label them. This old-school approach not only felt tedious — it also seemed like a missed opportunity to create a smarter, more delightful user experience.

Imagine if the website could instantly recognize whether a photo showed a dog or a cat, saving users time and making the site feel more modern and intuitive.

The goal of my project is to build an image classification model using Convolutional Neural Networks (CNNs) to automatically predict whether an uploaded image contains a dog or a cat — helping pet websites level up their onboarding process with a little help from deep learning.

### Project Outline

To find the best-performing model, I will start with the simplest approach: manually building a basic Convolutional Neural Network (CNN) using Keras and tuning it by hand.

Next, I will leverage Keras Tuner to automate the search for the best hyperparameters, allowing the model to improve through guided experimentation.

Finally, I will take advantage of transfer learning by using the well-established pre-trained model ResNet50V2 — building on the shoulders of giants to further boost performance and accelerate training.


### Table of Contents

1. [About the Data](#about-the-data)
2. [Custom CNN model](#custom-cnn-model)
3. [Tuning the mode](#tuning-the-model)
4. [Transfer Learning](#transfer-learning)
5. [Next Steps](#next-steps)
6. [References](#references)


### About the Data

The dataset originates from a Kaggle competition held in 2013 and consists of images of dogs and cats. These images were originally provided to Microsoft Research by Asirra, a project developed in partnership with Petfinder.com to help homeless pets find new homes.

The photos were manually classified by staff members at thousands of animal shelters across the United States, ensuring accurate labels for each image.

The dataset is separated in training data and test data, with the following charactertistics:

1. The train data contains 25,000 images: 12,500 dogs, 12,500 cats.
2. The train filenames include the label, such as "dog.8011.jpg"
3. The test data contains 391 images with an unknown number of cats and dogs.
4. The filename only contains the number used to identify the image, such as "9733.jpg"

To extract the label for each training image from its filename, I created a function called get_label that retrieves the label directly from the file path.

Because this function is called during the TensorFlow Dataset construction process, it uses TensorFlow operations such as tf.strings.split and tf.where. These TensorFlow-native methods are necessary to ensure compatibility with TensorFlow’s data pipelines, which are highly optimized for performance — enabling parallel processing and GPU acceleration during data preparation.

- tf.strings.split breaks a tensor string into parts, just like Python’s split(), but optimized to work inside TensorFlow graphs and pipelines.
- tf.where is similar to an "if-else" statement". It checks a condition, and based on whether it's True or False, it chooses between two options, in this case, it returns 1 if is the string contains 'dog' and 0 if it does not.

In addition to the test dataset, I created a validation dataset by splitting the train data into 80% train and 20% val. The validation set will be used to measure the performance of my models. 


### Custom CNN model

My baseline model is a simple Convolutional Neural Network (CNN) designed to provide an initial benchmark for classifying dog and cat images.

It follows a straightforward architecture:

**Rescaling Layer**: The input pixel values, originally in the range [0, 255], are scaled down to [0, 1] using Rescaling(1./255). This normalization helps improve training stability and convergence speed.

**Convolutional Layer**: A Conv2D layer with 32 filters of size 3×3 is applied, using stride 1 and 'same' padding. 'same' padding ensures that the output spatial dimensions match the input dimensions. The ReLU activation function introduces non-linearity, allowing the model to learn complex patterns.

**Flatten Layer**: The output from the convolutional layer is flattened into a 1D vector to prepare it for the dense (fully connected) layer.

**Dense Output Layer**: A Dense layer with units equal to the number of classes (len(class_names), i.e., 2 for dog and cat) and a softmax activation function, softmax ensures the output values are probabilities that sum to 1, making it suitable for multi-class classification.

**Compilation**:
The model is compiled with: 
- Optimizer: **Adam** — an adaptive learning rate optimizer that works well out of the box.
- Loss: **sparse_categorical_crossentropy** — appropriate since the class labels are integers (0 for cat, 1 for dog).
- Metric: **Accuracy** — to measure how often predictions match labels.


### Tuning the model


### Transfer learning


### Next Steps


### References
