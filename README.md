Digit Recognizer using Deep Learning and Dimensionality Reduction
This project focuses on recognizing hand-drawn digits from the famous MNIST dataset using deep learning techniques and dimensionality reduction methods. It combines Python libraries like Keras, Pandas, and Plotly for data manipulation, neural network modeling, and data visualization.

Code 1: Deep Learning with Keras
Data Loading: The code loads the MNIST dataset for both training and testing.
Data Preprocessing: It prepares the data by standardizing and reshaping the images.
Model Building: A simple feedforward neural network is constructed using Keras. It includes multiple layers, such as flattening, dense, and softmax.
Model Training: The model is trained using various optimization techniques. Different models are compared, including a basic feedforward model and a convolutional neural network (CNN).
Data Augmentation: Data augmentation is applied to improve model performance.
Model Evaluation: The model's performance is evaluated and used to make predictions on the test dataset.

Code 2: Dimensionality Reduction and Visualization
Feature Scaling: Standardization is applied to prepare the data.
Eigenvalue Decomposition: The eigenvectors and eigenvalues of the covariance matrix are calculated to perform dimensionality reduction.
Explained Variance Analysis: The code presents a plot showing the explained variance in the dataset and offers insights into dimensionality reduction.
Visualization: Visualizes the dataset using dimensionality reduction techniques like t-SNE and PCA.

Code 3: Multi-Layer Perceptron (MLP) for Digit Classification
Data Loading: The MNIST dataset is loaded.
Data Preprocessing: Data is scaled and standardized.
MLP Model: A multi-layer perceptron (MLP) model is defined with Keras. It consists of densely connected layers with activation functions and dropout layers.
Model Training: The MLP model is trained, and the accuracy is evaluated.
Test Predictions: The trained model makes predictions on the test data, which are then saved to a CSV file for submission.
This combined code demonstrates a comprehensive approach to digit recognition using deep learning and dimensionality reduction. It covers data preparation, model building, training, and evaluation, as well as visualization of data using dimensionality reduction techniques.

