# Human Activity Recognition Using Machine Learning and LSTM

## About the Project

This project is about Human Activity Recognition using the UCI Human Activity Recognition Using Smartphones dataset.

The main goal of the project is to predict what activity a person is doing based on sensor data collected from a smartphone.

The six activities in the dataset are:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

I worked on this project to understand how different machine learning models perform on the same dataset and to compare traditional machine learning models with an LSTM-based deep learning approach.

---

## Dataset

I used the UCI Human Activity Recognition Using Smartphones dataset.

The dataset contains data collected from 30 different subjects using smartphone accelerometer and gyroscope sensors.

The dataset is divided into:

- Training data: 7,352 samples
- Testing data: 2,947 samples
- 561 features

The 561 features are already engineered from the original sensor signals.

For the LSTM part of the project, I also worked with the raw sensor signals instead of using the 561 engineered features.

---

## What I Did in This Project

The project mainly follows these steps:

1. Loaded the dataset
2. Added feature names and activity labels
3. Checked the structure of the data
4. Checked for duplicate values
5. Checked for missing values
6. Performed exploratory data analysis
7. Studied the distribution of activities among subjects
8. Visualized different features
9. Used t-SNE to understand the feature space
10. Trained different machine learning models
11. Used GridSearchCV for hyperparameter tuning
12. Compared the models using different evaluation metrics
13. Built LSTM models using raw sensor data
14. Compared the LSTM results with the classical machine learning models

---

## Data Cleaning

Before training the models, I checked the dataset for basic data quality problems.

There were:

- 0 duplicate rows in the training data
- 0 duplicate rows in the testing data
- 0 missing/null values in the training data
- 0 missing/null values in the testing data

The activity distribution was also checked to understand how many samples were available for each activity.

---

## Exploratory Data Analysis

I performed different types of analysis to understand the dataset before building the models.

Some of the things I looked at were:

- Number of samples for each activity
- Activity distribution for different subjects
- Sensor-based features
- Acceleration magnitude
- Gravity-related features
- Differences between stationary and moving activities

I also used t-SNE to visualize the 561-dimensional feature space and see how the different activities are distributed.

---

# Machine Learning Models

For the first part of the project, I used the 561 engineered features provided by the dataset.

I tested the following models:

- Logistic Regression
- Linear SVC
- RBF SVM
- Decision Tree
- Random Forest

I used GridSearchCV to find better hyperparameters for the models.

## Results

| Model | Accuracy |
|---|---:|
| Linear SVC | 96.67% |
| Logistic Regression | 96.54% |
| RBF SVM | 96.27% |
| Random Forest | 91.45% |
| Decision Tree | 86.22% |

The best-performing classical machine learning model was Linear SVC with an accuracy of 96.67%.

---

# LSTM Models

After working with the traditional machine learning models, I wanted to try a deep learning approach.

For this part, I used the raw sensor signals instead of the 561 engineered features.

The input to the LSTM models was based on:

- 128 time steps
- 9 sensor channels
- 6 activity classes

I experimented with different LSTM architectures to see how changing the model affected the results.

### LSTM Model 1

A single LSTM layer with 32 units.

Accuracy:

**88.56%**

### LSTM Model 2

A two-layer LSTM with 48 units followed by 32 units.

Accuracy:

**92.16%**

This was the best-performing LSTM model in my experiments.

### LSTM Model 3

A two-layer LSTM with 64 units followed by 48 units.

Accuracy:

**88.77%**

---

## LSTM Results

| Model | Accuracy |
|---|---:|
| LSTM (32) | 88.56% |
| LSTM (48, 32) | 92.16% |
| LSTM (64, 48) | 88.77% |

The two-layer LSTM with 48 and 32 units performed the best among the LSTM models.

---

# Final Comparison

The best results from both approaches were:

| Approach | Best Model | Accuracy |
|---|---|---:|
| Classical Machine Learning | Linear SVC | 96.67% |
| Deep Learning | LSTM (48, 32) | 92.16% |

In my experiments, the classical machine learning approach performed better than the LSTM approach on this dataset.

This was an interesting result because I initially expected the LSTM to perform better since it can work with sequential sensor data. However, the 561 engineered features provided by the dataset already contain a lot of useful information, which helped the classical models achieve very high accuracy.

---

## What I Learned

Through this project, I learned how to work with a real-world machine learning dataset from start to finish.

Some of the main things I learned were:

- How to explore sensor data
- How to visualize data and understand patterns
- How to train different machine learning models
- How GridSearchCV can be used for hyperparameter tuning
- How to evaluate classification models
- How confusion matrices can help understand model mistakes
- How LSTM models can be used for time-series sensor data
- The difference between using engineered features and raw sequential data
- Why comparing multiple models is important instead of assuming that a more complex model will always perform better

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

---

## Project Structure

```text
Human-Activity-Recognition/
│
├── UCI_HAR_Dataset/
│
├── MLPROJECT1.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn tensorflow
```

Download the UCI HAR dataset and place it inside the project folder.

Then open:

```text
MLPROJECT1.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

---

## Conclusion

This project helped me understand the complete machine learning workflow, from exploring and cleaning the data to training and comparing different models.

The best result I achieved was **96.67% accuracy using Linear SVC**. The best LSTM model achieved **92.16% accuracy**.

The main takeaway for me was that a more complex deep learning model does not always give better results. The quality and representation of the features can be just as important as the model itself.
