<div style="text-align: center;">
    <i><b>VIETNAM NATIONAL UNIVERSITY, HO CHI MINH CITY</b></i>
    <br>
    <i><b>UNIVERSITY OF SCIENCE</b></i>  
    <br>
    <i><b>FACULTY OF ADVANCED INFORMATION TECHNOLOGY</b></i>
    <br>
    <b>------------o------------</b>
    <br> 
    <b>INTRODUCTION TO MACHINE LEARNING - CSC14005</b>
</div>

<h1 style="text-align: center; color: red;"><b>Course Project</b></h1>
<h1 style="text-align: center; color: red;"><b>Neural Networks for Classification</b></h1>

<div style = "text-align: center;"> <ins><b>Last updated date:</b></ins> 27 / 12 / 2024</div>

Neural networks are powerful tools for classification tasks. In this course project, students will explore, experiment with different neural network libraries and frameworks for classification tasks.

The focus will be on implementing across various libraries and frameworks as following:
1. **Scikit-learn (MLPClassifier)**
2. **TensorFlow/Keras**
3. **PyTorch**

Students are required to conduct a detailed comparison of the above libraries and frameworks based on the following criteria:
1. **Training Time**
2. **GPU Memory Usage**
3. **Pros and Cons** (derived from your research)
4. **Ease of use** (based on your personal experience throughout the task)

By completing this project, students will gain experience with multiple libraries and frameworks for designing and training neural networks, as well as insights into the strengths and limitations of these tools for future classification tasks.

## **1. Dataset**
* For this project, students will use the CIFAR-10, a widely dataset for image classification tasks. The objective is to classify images into their corresponding classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck).
* The dataset contains $60000$ $32 \times 32$ RGB images ($50000$ for training and $10000$ for testing).
* Visit: http://www.cs.toronto.edu/~kriz/cifar.html

## **2. Specification**
* Building a classifier model involves several stages. Regardless of the libraries or frameworks used, the basic procedure is consistent and follows these steps:
### **2.1. Data Preparation**
* The first step in building any classifier model is preparing the data:
  * **Data Collection:** Gather the data from a dataset.
  * **Data Preprocessing:** This includes removing missing data, encoding categorical features, normalizing the data, and splitting the data into training, validation, and test sets.
  * **Feature Engineering:** Features may need to be engineered to improve performance.
  
  *Notes: In this project, students are required to (1) normalize and (2) flatten the image data; then (3) apply one-hot encoding to the output labels.*
### **2.2. Model Design**
* Designing the model is defining the architecture of the classifier. For example, in case of MLP:
  * **Number of layers:** Decide on how many hidden layers the MLP will have.
  * **Number of neurons per layer:** Decide on the number of neurons in each layer.
  * **Activation functions:** Sigmoid, ReLU and Softmax (for multi-class classification).
  * **Dropout or batch normalization:** If needed, incorporate dropout or batch normalization to prevent overfitting and to speed up training phase.
* To build a MLP classifier, the following libraries and frameworks can be considered:
  * Scikit-learn (MLPClassifier).
  * TensorFlow/Keras.
  * PyTorch.

*Notes: In this project, students need to use the same architecture when experimenting and evaluating across the libraries and frameworks to ensure consistency and fair comparison.*
### **2.3. Selecting Loss Function and Optimizer**
* The next step is selecting the appropriate loss function and optimizer:
  * **Loss function:** For classification tasks, the common loss functions are cross-entropy loss (for multi-class problems) and binary cross-entropy loss (for binary classification).
  * **Optimizer:** such as SGD, Adam, etc. Adjust the learning rate to control the step size.
### **2.4 Model Training**
* Once the model architecture, loss function, and optimizer are defined, proceed with training model:
  * **Epochs:** Specify the number of epochs (iterations over the entire training dataset).
  * **Batch size:** Choose the batch size, which defines the number of samples per gradient update.
  * **Training process:** For each batch, perform forward propagation, compute the loss, and update the model weights using backpropagation.
  * **Early stopping:** Monitor the validation set performance during training process and stop if it doesn’t improve after a set number of epochs.
  
*Notes: In this project, students need to record the training time and the GPU consumption of libraries and frameworks during the training process for later comparison.*
### **2.5. Model Evaluation**
* After training, it is important to evaluate the performance of the model on unseen data (test set):
  * **Evaluation metrics:** Accuracy, precision, recall, F1-score, etc., depending on the problem.
  * **Confusion matrix:** used for analyzing how well the model performs on each class.
### **2.6. Futher Usage**
* If the model meets the required quality standards, it can be prepared for practical applications by (1) saving and loading model, and (2) deploy the model.

*Notes: In this project, students are NOT required to deploy the model as a real-life product; however, you are encouraged to explore it on your own to enhance your skills.*

## **3. Work Assignment Table**
| Student ID |   Full Name   |               General Tasks               | Detailed Tasks                  | Completion |
|:----------:|:-------------:|:-----------------------------------------:| ------------------------------- |:----------:|
|  22127147  |  Đỗ Minh Huy  |           **Data Preparation**            | Data Collection                 |    100%    |
|            |               |                                           | Data Preprocessing              |    100%    |
|            |               |                                           | Data Normalization              |    100%    |
|            |               |                                           | Data Flatten                    |    100%    |
|            |               |                                           | Data Encoding                   |    100%    |
|            |               |             **Model Design**              | Number of layers                |    100%    |
|            |               |                                           | Number of neurons per layer     |    100%    |
|            |               |                                           | Activation Functions            |    100%    |
|            |               |                                           | Drop out or batch normalization |    100%    |
|            |               | **Selecting Loss Function and Optimizer** | Loss Function                   |    100%    |
|            |               |                                           | Optimizer                       |    100%    |
|  22127322  | Lê Phước Phát |           **Data Preparation**            | Data Visualization              |    100%    |
|            |               |                                           | Scikit-learn                    |    100%    |
|            |               |            **Model Training**             | Tensorflow/Keras                |    100%    |
|            |               |                                           | Pytorch                         |    100%    |
|            |               |           **Model Evaluation**            | Each Model Evaluation           |    100%    |
|            |               |                                           | Compare and Analysis            |    100%    |
|            |               |             **Further Usage**             | Save and Load Model             |    100%    |
|            |               |                                           | Deploy The Model                |    100%    |
|            |               |                **Report**                 |                                 |    100%    |

## **4. Self-evaluation of the assignment requirements**
| No. |        Detailed Tasks        | Completion Rate |
|:---:|:----------------------------:|:---------------:|
|  1  | Scikit-learn (MLPClassifier) |     $100\%$     |
|  2  |      Tensorflow / Keras      |     $100\%$     |
|  3  |           Pytorch            |     $100\%$     |
|  4  |            Report            |     $100\%$     |