# ♻️ Waste Image Classification using CNN

A deep learning-based computer vision project that classifies waste images into six different categories using a Convolutional Neural Network (CNN).

The project covers the complete machine learning workflow:

**Dataset → EDA → Data Preparation → Data Augmentation → CNN → Model Evaluation → Model Saving → Streamlit Deployment**

The final CNN model achieved **80.26% test accuracy**.

---

## 📌 Project Overview

Waste segregation is an important part of recycling and waste management.

Different types of waste require different recycling and disposal processes. This project explores how computer vision and deep learning can be used to automatically classify waste images into different categories.

The model takes an image as input and predicts one of the following six waste categories:

- 📦 Cardboard
- 🍾 Glass
- 🔩 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

The trained model is integrated with a Streamlit application where users can upload an image and receive a predicted waste category along with the model's confidence.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze the waste image dataset.
- Perform exploratory data analysis.
- Prepare images for CNN training.
- Resize and normalize images.
- Handle class imbalance using class weights.
- Apply image augmentation.
- Build a CNN image classification model.
- Reduce overfitting.
- Evaluate the model using multiple metrics.
- Save the trained model.
- Build a Streamlit application for prediction.

---

## 🌍 Why This Project Matters

Automated waste classification can support applications such as:

- Smart waste segregation
- Automated recycling systems
- Smart waste bins
- Waste sorting systems
- Recycling facilities
- Environmental monitoring
- Computer vision-based waste management

This project also demonstrates how a deep learning model can be developed from raw image data and converted into a working application.

---

## ❓ Problem Statement

Given an image of a waste item, the goal is to develop a deep learning model that automatically predicts which waste category the image belongs to.

### Input

An image of a waste item.

### Output

One of six waste categories:

```text
Cardboard
Glass
Metal
Paper
Plastic
Trash

📊 Dataset

This project uses the TrashNet dataset.

Dataset source:

https://github.com/garythung/trashnet

The dataset contains 2,527 images belonging to six waste categories.

Category	Number of Images
Cardboard	403
Glass	501
Metal	410
Paper	594
Plastic	482
Trash	137
Total	2,527

The dataset is imbalanced because different categories contain different numbers of images.

🔎 Exploratory Data Analysis

The project begins with exploratory data analysis to understand the dataset.

The EDA includes:

Number of images in each class
Class distribution
Sample image visualization
Image inspection
Identification of class imbalance

The analysis showed that the number of images varies considerably between categories.

🛠️ Data Preparation

All images were prepared for CNN training.

⚖️ Handling Class Imbalance

The dataset contains different numbers of images for each category.

🔄 Data Augmentation

Data augmentation was applied to improve model generalization and reduce overfitting.

The following techniques were used:

Random horizontal flip
Random rotation
Random zoom
Random contrast

🧠 CNN Model

The final CNN consists of multiple convolutional blocks.

Architecture
Input Image
     ↓
Data Augmentation
     ↓
Rescaling
     ↓
Conv2D - 32 filters
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Conv2D - 64 filters
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Conv2D - 128 filters
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Conv2D - 256 filters
     ↓
Batch Normalization
     ↓
Max Pooling
     ↓
Global Average Pooling
     ↓
Dense - 128
     ↓
Dropout
     ↓
Dense - 6
     ↓
Softmax
     ↓
Prediction

🧩 CNN Components
Convolutional Layers

Convolutional layers learn visual features such as:

Edges
Shapes
Textures
Patterns
Object structures
Batch Normalization

Batch normalization helps stabilize the training process.

Max Pooling

Max pooling reduces the spatial dimensions of feature maps while preserving important features.

Global Average Pooling

Global average pooling reduces the number of parameters compared with a large Flatten layer and helps reduce overfitting.

Dropout

Dropout helps prevent overfitting by randomly disabling neurons during training.

Softmax

The final softmax layer produces probabilities for the six waste categories.

🏋️ Model Training

The final CNN was trained using:

Configuration	Value
Optimizer	Adam
Initial Learning Rate	0.001
Loss Function	Sparse Categorical Crossentropy
Metric	Accuracy
Maximum Epochs	30
Batch Size	32
Class Weights	Yes
Early Stopping	Yes
Reduce Learning Rate	Yes
Model Checkpointing	Yes
🛑 Early Stopping

Early stopping was used to reduce overfitting and prevent unnecessary training.

The model monitored validation loss and restored the best model weights.

📉 Learning Rate Reduction

ReduceLROnPlateau was used to reduce the learning rate when validation loss stopped improving.

This allows the model to make smaller parameter updates during later stages of training.

📊 Baseline Model

An initial CNN model was developed as a baseline.

Baseline Result
Test Accuracy: 64.84%

The baseline model showed noticeable overfitting, with training accuracy significantly higher than validation performance.

🚀 Improved CNN Model

The final model introduced several improvements:

Data augmentation
Batch normalization
Global average pooling
Dropout
Class weighting
Early stopping
Learning-rate reduction
Model checkpointing
Final Result
Test Accuracy: 80.26%
Accuracy Improvement
Baseline CNN : 64.84%
Final CNN    : 80.26%

The final model improved test accuracy by approximately 15.42 percentage points compared with the baseline model.

📈 Model Evaluation

The final model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion matrix
Classification report
🏆 Final Performance
Metric	Score
Test Accuracy	80.26%
Macro Precision	78.66%
Macro Recall	82.26%
Macro F1-Score	79.27%
Weighted F1-Score	80.58%
📋 Classification Report
              precision    recall  f1-score   support

   cardboard     0.9333    0.9180    0.9256        61
       glass     0.8966    0.6933    0.7820        75
       metal     0.6712    0.7903    0.7259        62
       paper     0.8916    0.8315    0.8605        89
     plastic     0.7714    0.7500    0.7606        72
       trash     0.5556    0.9524    0.7018        21

    accuracy                         0.8026       380
   macro avg     0.7866    0.8226    0.7927       380
weighted avg     0.8220    0.8026    0.8058       380
🔀 Confusion Matrix Analysis

The confusion matrix was used to understand the classification errors.

Some visually similar categories were difficult for the model to distinguish.

Major confusion patterns included:

Glass → Plastic
Glass → Metal
Plastic → Metal
Metal → Trash

These errors can occur because different waste materials can have similar:

Shapes
Colors
Textures
Reflective surfaces
Backgrounds

After applying class weighting, the model achieved high recall for the trash category.

💾 Final Model

The final trained model was saved as:

models/waste_cnn_final.keras

The model can be loaded using:

import tensorflow as tf

model = tf.keras.models.load_model(
    "models/waste_cnn_final.keras"
)
🌐 Streamlit Application

The trained CNN was deployed using Streamlit.

The application allows users to:

Upload a waste image.
Preview the uploaded image.
Click the Classify Image button.
Receive the predicted waste category.
View the model's prediction confidence.
Supported Categories
Cardboard
Glass
Metal
Paper
Plastic
Trash
🔄 Application Workflow
User
  ↓
Upload Image
  ↓
Convert to RGB
  ↓
Resize to 128 × 128
  ↓
Convert to NumPy Array
  ↓
CNN Model
  ↓
Class Probabilities
  ↓
Highest Probability
  ↓
Predicted Waste Category
📁 Project Structure
Waste_Image_Classification_CNN/
│
├── data/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── notebooks/
│   ├── 01_Waste_Dataset_EDA.ipynb
│   ├── 02_Data_Preparation.ipynb
│   └── 03_CNN_Model.ipynb
│
├── models/
│   └── waste_cnn_final.keras
│
├── src/
│   ├── __init__.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
📓 Notebook Description
01_Waste_Dataset_EDA.ipynb

Contains:

Dataset exploration
Class distribution
Image visualization
Dataset analysis
Initial observations
02_Data_Preparation.ipynb

Contains:

Dataset loading
Image resizing
Dataset splitting
TensorFlow dataset creation
Image normalization
Dataset optimization
03_CNN_Model.ipynb

Contains:

Baseline CNN
Baseline evaluation
Data augmentation
Class weights
Improved CNN
Model training
Early stopping
Learning-rate reduction
Model evaluation
Confusion matrix
Classification report
Final model saving
🛠️ Technologies Used
Programming Language
Python 3.10
Deep Learning
TensorFlow
Keras
Convolutional Neural Networks
Data Processing
NumPy
Pandas
Pillow
Machine Learning
Scikit-learn
Visualization
Matplotlib
Seaborn
Deployment
Streamlit
Development Tools
Visual Studio Code
Jupyter Notebook
Git
GitHub
📦 Installation

Clone the repository:

git clone https://github.com/greeshma078/Waste_Image_Classification_CNN.git

Navigate to the project directory:

cd Waste_Image_Classification_CNN

Install the required dependencies:

pip install -r requirements.txt
▶️ Run the Streamlit Application

Run the following command from the project directory:

streamlit run app.py

The Streamlit application will open in your web browser.

🧪 Making Predictions

The prediction logic is implemented in:

src/predict.py

The prediction pipeline:

Loads the trained CNN model.
Converts the uploaded image to RGB.
Resizes the image to 128 × 128.
Converts the image into a NumPy array.
Adds the batch dimension.
Passes the image through the CNN.
Obtains class probabilities.
Selects the class with the highest probability.
Displays the predicted category and confidence.
🧠 Key Learnings

This project demonstrates practical knowledge of:

Computer Vision
Image Classification
Convolutional Neural Networks
TensorFlow and Keras
Image preprocessing
Image normalization
Data augmentation
Batch normalization
Dropout
Global average pooling
Class imbalance
Class weighting
Train-validation-test splitting
Early stopping
Learning-rate scheduling
Model checkpointing
Confusion matrix
Classification report
Model saving and loading
Streamlit deployment
Git and GitHub
⚠️ Limitations

The project has some limitations.

Dataset Size

The dataset contains 2,527 images, which limits the variety of examples available for training.

Class Imbalance

The number of images differs significantly between categories.

Similar Waste Materials

Some categories are visually similar, making classification difficult.

For example:

Glass ↔ Plastic
Metal ↔ Plastic
Metal ↔ Trash
Real-World Conditions

Real-world images may contain:

Different lighting
Complex backgrounds
Multiple objects
Different camera angles
Occluded objects
Different object sizes

Therefore, performance on new real-world images may differ from the test-set performance.

🚀 Future Improvements

Possible future improvements include:

1. Transfer Learning

Use pretrained models such as:

MobileNetV2
EfficientNet
ResNet
MobileNet
2. Larger Dataset

Collect additional real-world waste images.

3. Advanced Data Augmentation

Experiment with:

Random brightness
Random translation
Random cropping
Additional image transformations
4. Real-Time Waste Detection

Extend the project to support camera-based real-time classification.

5. Smart Waste Bin

The model could potentially be integrated with a camera and hardware system:

Camera
   ↓
CNN Model
   ↓
Waste Classification
   ↓
Microcontroller
   ↓
Automatic Sorting

📌 Project Highlights
Dataset              : TrashNet
Total Images         : 2,527
Number of Classes    : 6
Image Size           : 128 × 128
Model                : CNN
Data Augmentation    : Yes
Class Weighting      : Yes
Baseline Accuracy    : 64.84%
Final Accuracy       : 80.26%
Deployment           : Streamlit
Model Format         : .keras
👩‍💻 Author

Greeshma Reddy

B.Tech – Artificial Intelligence & Data Science

Areas of Interest
Data Science
Machine Learning
Deep Learning
Computer Vision
Natural Language Processing
Generative AI

⭐ Conclusion

This project demonstrates an end-to-end computer vision workflow for classifying waste images using a Convolutional Neural Network.

Starting from the TrashNet dataset, the project covers exploratory data analysis, preprocessing, image augmentation, class-imbalance handling, CNN development, model evaluation, model saving, and Streamlit deployment.

The final model achieved 80.26% test accuracy, demonstrating the application of deep learning to automated waste image classification while also highlighting challenges related to limited data, class imbalance, and visually similar waste categories.
