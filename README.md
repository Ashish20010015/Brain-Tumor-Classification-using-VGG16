# Brain-Tumor-Classification-using-VGG16
In this machine learning project, we build a classifier to detect the brain tumor (if any) from the MRI scan images. By now it is evident that this is a binary classification problem. Examples of such binary classification problems are Spam or Not spam, Credit card fraud (Fraud or Not fraud).

I am using the dataset from https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
The images are split into two folders yes and no each containing images with and without brain tumors respectively. There are a total of 253 images.

**Tools and Libraries used**
Brain tumor detection project uses the below libraries and frameworks:
Python, TensorFlow, Keras, Numpy, Scikit-learn, Matplotlib, OpenCV

**Steps to Develop Brain Tumor Classifier**
Our approach to building the classifier is discussed in the steps:

1. Perform Exploratory Data Analysis (EDA) on brain tumor dataset
2. Build a CNN model
3. Train and Evaluate our model on the dataset

**Step 1. Perform Exploratory Data Analysis (EDA)**
The brain tumor dataset contains 2 folders “no” and “yes” with 98 and 155 images each. Load the folders containing the images to our current working directory. Using the imutils module, we extract the paths for all the images and store them in a list called **image_paths**.

Now, I iterate over each of the paths and extract the directory name (no or yes in our case which acts as the label), and resize the image size to **224×224 pixels**. The **imread()** function of the **cv2** module converts brain tumor images to pixel information.

As you can see, we have stored the image and its respective label in lists. But the labels are strings which can’t be interpreted by machines. So, apply **One-hot encoding** to the labels.Also, normalize the images and convert our lists to numpy arrays to further split our dataset.

let’s split the dataset into training and testing sets in the ratio of **9-1** using the **train_test_split()** function in the **Scikit-learn package**.

**Step 2: Build a CNN Model**
A Convolutional Neural Network or CNN for short is a deep neural network widely used for analyzing visual images. These types of networks work well for tasks like image classification and detection, image segmentation. There are 2 main parts of a CNN:

1. A convolutional layer that does the job of feature extraction
2. A fully connected layer at the end that utilizes the output of the convolutional layers and predicts the class of the image.

**TensorFlow** provides **ImageDataGenerator** which is used for data augmentation. Data Augmentation is extremely helpful in cases where the input data is very less. So we use different transformations to increase the dataset size. It provides various transformations like rotation, flipping images horizontally, vertically, zoom, etc.
We are using the transformations fill_mode and rotation_range to fill the out of boundary pixels with the pixel **“nearest”** to them and include a rotation of **15 degrees** to the images.

we utilize the power of **Transfer Learning** to make best predictions. Transfer learning is about leveraging feature representations from a pre-trained model, so you don’t have to train a new model from scratch.

we are using the **VGG16 state-of-the-art network model**. There are a number of pre-trained models available for use in Keras. Freeze the layers of our model. By doing this, the network is not trained from the very beginning. It uses the weights of previous layers and continues training for the layers we added on top of those layers. This reduces the training time by a drastic amount.

Now build the model and compile it using the **Adam as optimizer** with a **learning rate of 0.001** and **accuracy as metric**. As we are building a binary classifier and the input is an image, **binary cross entropy** is used as a **loss function**.

**Step 3: Train and evaluate the model**
The model is trained on **10 epochs** (full iterations) with train_steps for training set and validation_steps for validation set in each epoch. The **batch size** for each epoch is taken as **8**.






