![cat_dog_banner](https://github.com/user-attachments/assets/d6340aa7-1402-4a42-86c1-0f36d7150668)


# Cats vs Dogs
### Image Classification using Convolutional Neural Network

Having had both dogs and cats in our family for many years, I can confidently tell the difference between the two. Yet, many pet-focused websites still struggle to distinguish between them when users are setting up profiles for their pets.

While booking a stay for our pets before a family vacation, I was asked to upload photos of each pet one by one and manually label them. This old-school approach not only felt tedious — it also seemed like a missed opportunity to create a smarter, more delightful user experience.

Imagine if the website could instantly recognize whether a photo showed a dog or a cat, saving users time and making the site feel more modern and intuitive.

The goal of my project is to build an image classification model using Convolutional Neural Networks (CNNs) to automatically predict whether an uploaded image contains a dog or a cat — helping pet websites level up their onboarding process with a little help from deep learning.

# Project Outline

To find the best-performing model, I will start with the simplest approach: manually building a basic Convolutional Neural Network (CNN) using Keras and tuning it by hand.

Next, I will leverage Keras Tuner to automate the search for the best hyperparameters, allowing the model to improve through guided experimentation.

Finally, I will take advantage of transfer learning by using the well-established pre-trained model ResNet50V2 — building on the shoulders of giants to further boost performance and accelerate training.


# Table of Contents

1. [About the Data](#about-the-data)
2. [Custom CNN model](#custom-cnn-model)
3. [Keras Tuner](#keras-tuner)
4. [Transfer Learning](#transfer-learning)
5. [Conclusion](#conclusion)
6. [References](#references)


# About the Data

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


# Custom CNN model
### Baseline model

My baseline model is a simple Convolutional Neural Network (CNN) designed to provide an initial benchmark for classifying dog and cat images. It follows a straightforward architecture:

**Rescaling Layer**: The input pixel values, originally in the range [0, 255], are scaled down to [0, 1] using Rescaling(1./255). This normalization helps improve training stability and convergence speed.

**Convolutional Layer**: A Conv2D layer with 32 filters of size 3×3 is applied, using stride 1 and 'same' padding. 'same' padding ensures that the output spatial dimensions match the input dimensions. The ReLU activation function introduces non-linearity, allowing the model to learn complex patterns.

**Flatten Layer**: The output from the convolutional layer is flattened into a 1D vector to prepare it for the dense (fully connected) layer.

**Dense Output Layer**: A Dense layer with units equal to the number of classes (len(class_names), i.e., 2 for dog and cat) and a softmax activation function, softmax ensures the output values are probabilities that sum to 1, making it suitable for multi-class classification.

**Compilation**:
The model is compiled with: 
- Optimizer: **Adam** — an adaptive learning rate optimizer that works well out of the box.
- Loss: **sparse_categorical_crossentropy** — appropriate since the class labels are integers (0 for cat, 1 for dog).
- Metric: **Accuracy** — to measure how often predictions match labels.

### Tuning the model by hand:

After tuning the model by hand, my best model builds upon the baseline by introducing a deeper architecture with multiple convolutional and pooling layers, along with dropout for regularization.

The architecture includes three convolutional layers are stacked, each with 64 filters of size 3×3, stride 1, 'same' padding, and ReLU activation.
Each convolutional layer is followed by a Dropout layer that randomly drops 20% of neurons to reduce overfitting, as well as a MaxPooling2D layer, which downsamples feature maps using a pool size of 3×3 to progressively reduce spatial dimensions.

The best metrics were seen at epoch 9, and yielded:

- train loss: 0.2042 
- train accuracy: 0.9160
- val_loss: 0.3702
- val_accuracy: 0.8362

<img src="https://github.com/user-attachments/assets/6fb73a2c-9867-42c9-92c0-d51edf131b20" width="400">
<img src="https://github.com/user-attachments/assets/e0053198-b7f2-40f3-b4c8-08497a67eba3" width="400">

Based on the above plots, although the training loss and accuracy continue to improve steadily across epochs, the validation performance only improves slightly and shows consistent fluctuations.

This indicates that the model is struggling to make meaningful improvements on unseen data. While the model fits the training data increasingly well, the validation metrics remain unstable and eventually stagnate. This clear gap between training and validation performance suggests that the model is overfitting — learning the training examples too well without generalizing effectively to new data.


# Keras Tuner

Next, I used the keras tuner to help me search across many combinations of hyperparameters to find the best performing model. 

To optimize the model’s architecture, I implemented a custom tune_model function for Keras Tuner. This function dynamically builds a CNN model based on different hyperparameters selected during the tuning process:

Number of Convolutional Layers: Tuned between 1 and 3 layers.

Convolutional Layer Parameters: 
- Filters: Either 64 or 128 filters per layer.
- Kernel Size: Either 3×3 or 4×4.
- Stride: 1 or 2 pixels.
- Activation Function: Either ReLU or Tanh.

Dropout Rate: Applied after each convolutional layer. Tuned between 0.0 and 0.3 (in increments of 0.1) to control overfitting.
Pooling: After each convolutional block, a MaxPooling2D layer with a 3×3 pool size is applied.
Final Layers: A Flatten layer transforms the output into a 1D vector. A Dense layer with a softmax activation outputs the final classification.
Compilation: Optimizer: Adam. Loss: Sparse categorical crossentropy. Metric: Accuracy

This setup allows Keras Tuner to systematically explore different combinations of model depth, convolutional parameters, and regularization strategies to find the most effective architecture for the task.
My best model from the keras tuner was found on trial #18 out of 22 completed trials. The best model summary:

<img width="479" alt="Screenshot 2025-04-20 at 12 16 54 PM" src="https://github.com/user-attachments/assets/a0bae699-a716-404d-99a4-d5f8938c5a29" />

<img src="https://github.com/user-attachments/assets/20231e36-91ae-4c26-a6a4-041e8f392dba" width="400">


With the following hyperparameters:

- 'num_layers': 3,
- 'layer_1_units': 64,
- 'layer_1_kernel_size': 3,
- 'layer_1_stride': 1,
- 'layer_1_activation': 'relu',
- 'layer_1_dropout': 0.2,

<img src="https://github.com/user-attachments/assets/3c31b94a-d554-4a1e-a66e-3039eace6fa6" width="400">
<img src="https://github.com/user-attachments/assets/969965f5-cc11-4fad-9e12-27378e663eb9" width="400">

The above plots show gradual improvement in both training and validation performance up to around epoch 7, after which the validation metrics begin to diverge and decline. While the gap between training and validation performance starts out small, it continues to widen as training progresses. This growing gap indicates that overfitting is still a significant issue in the model.

Next, I will use transfer learning to build upon pre-trained models.


# Transfer learning

Transfer learning allows me to start with a strong, proven model that already understands basic patterns in images — so I can focus on fine-tuning it for my specific task, faster and with better results. Pre-trained models have already learned very strong, general features. Even if my dataset is small, I can get high-quality feature extraction immediately. Results are often much better than training a small CNN from scratch.

In my transfer learning model, I leveraged the keras application ResNet50V2. I chose ResNet50V2 for its smaller size and faster time (ms) per inference step on both CPU and GPU. 

For my base model I loaded the ResNet50V2 model pretrained on ImageNet, excluding its top classification layers (include_top=False). I then froze the base model, so that the the pretrained ResNet50V2 weights will not be trained in order to preserve the powerful feature representations it already learned. To preprocess the data, I applied ResNet50V2’s recommended preprocess_input function to scale and normalize the input images correctly before feeding them into the model.

I started with a simple transfer model, but immediately saw a significant improvement in performance compared to my CNN models trained from stratch. I then tuned the hyperparameters by hand, and reached my best model, which contains 2 Dense layers with 32 units each and ReLU activation, each followed by Dropout layers for regularization.

<img src="https://github.com/user-attachments/assets/7b0bc7a8-636f-4afc-bf8f-d431e8308701" width="400">
<img src="https://github.com/user-attachments/assets/c635a4a3-b7bc-4a92-9c2b-4552f2c94115" width="400">

The above plots show that both train and loss are improving up to the third epoch, at which point val starts to diverge. Regardless, it is definitely my best model by far as can be seen in the below metrics comparisons:

- Manually Tuned Model: loss: 0.2042 - accuracy: 0.9160 - val_loss: 0.3702 - val_accuracy: 0.8362
- Keras Tuner Model: loss: 0.2055 - accuracy: 0.9155 - val_loss: 0.3302 - val_accuracy: 0.8628
- Transfer Model: loss: 0.0468 - accuracy: 0.9847 - val_loss: 0.0766 - val_accuracy: 0.9824


# Conclusion

This project lays the groundwork for building robust image classification models using both custom CNNs and transfer learning strategies. Through model design, hyperparameter tuning, and leveraging pretrained architectures like ResNet50V2, significant progress was made toward accurately distinguishing between dogs and cats.

However, this is only the beginning! 

For my steps steps, I would like to explore the following improvements to my model:
  1. Applying Keras Tuner to the transfer learning model to systematically optimize the dense layers, dropout rates, and learning rates.
  2. Enhancing regularization techniques and expanding the dataset to further improve generalization and resilience against overfitting.


# References

 1. Will Cukierski. Dogs vs. Cats. https://kaggle.com/competitions/dogs-vs-cats, 2013. Kaggle.
 2. Microsoft Research. Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization. Dataset originally available through Petfinder.com and hosted via Kaggle.
 3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *European Conference on Computer Vision (ECCV)*. https://arxiv.org/abs/1603.05027
 4. TensorFlow Authors. (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. https://www.tensorflow.org/
 5. Banner Image credited to FreeVector.com: <a href="https://www.freevector.com/cute-cat-and-dogs-background-81873">FreeVector.com</a>
