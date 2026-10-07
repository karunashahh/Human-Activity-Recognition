# Human Activity Recognition Using Machine Learning and LSTM

A machine learning project for classifying human activities using smartphone sensor data.

The project compares **classical machine learning models using engineered features** with **LSTM neural networks trained on raw sensor sequences**. The goal was to see how different approaches perform on the same Human Activity Recognition problem.

The experiments use the **UCI Human Activity Recognition Using Smartphones Dataset**, which contains sensor measurements collected from smartphones while participants performed different activities.

---

## Project Overview

Human Activity Recognition (HAR) is the task of identifying what a person is doing based on sensor data.

In this project, six activities are classified:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

Two different approaches were compared:

1. **Classical Machine Learning**
   - Logistic Regression
   - Linear SVC
   - RBF SVM
   - Decision Tree
   - Random Forest

2. **Deep Learning**
   - LSTM models trained directly on raw sensor sequences

The main objective was not just to train one model, but to compare different approaches and understand which type of model works better for this dataset.

---

## Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains:

- **7,352 training samples**
- **2,947 test samples**
- **561 engineered features**
- **6 activity classes**

The sensor data was collected from smartphone accelerometers and gyroscopes.

The activities are:

| Activity | Description |
|---|---|
| WALKING | Walking normally |
| WALKING_UPSTAIRS | Walking upstairs |
| WALKING_DOWNSTAIRS | Walking downstairs |
| SITTING | Sitting |
| STANDING | Standing |
| LAYING | Lying down |

The original dataset already provides a predefined training and testing split, which was used in this project.

---

## Project Workflow

The overall workflow was:

```text
UCI HAR Dataset
       |
       v
Data Loading
       |
       v
Data Cleaning & Inspection
       |
       v
Exploratory Data Analysis
       |
       +-----------------------------+
       |                             |
       v                             v
Engineered Features             Raw Sensor Data
       |                             |
       v                             v
Classical ML Models                LSTM Models
       |                             |
       +-------------+---------------+
                     |
                     v
              Model Comparison
                     |
                     v
               Final Analysis
```

---

## Data Exploration

Before training the models, the dataset was inspected to understand its structure and distribution.

The analysis included:

- Checking the shape of training and testing data
- Checking for missing values
- Checking for duplicate rows
- Examining activity distribution
- Comparing stationary and moving activities
- Exploring acceleration magnitude
- Examining gravity-related features
- Visualizing the data using plots
- Applying **t-SNE** for feature-space visualization

The dataset contains no missing values in the examined training and testing data.

---

## Classical Machine Learning

The first set of experiments used the **561 engineered features** provided by the UCI HAR dataset.

The following models were trained and evaluated:

- Logistic Regression
- Linear SVC
- RBF SVM
- Decision Tree
- Random Forest

Hyperparameters were tuned using **GridSearchCV** before evaluating the models on the test set.

---

## Classical ML Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 96.54% | 96.71% | 96.48% | 96.54% |
| **Linear SVC** | **96.67%** | **96.96%** | **96.63%** | **96.70%** |
| RBF SVM | 96.27% | 96.43% | 96.14% | 96.23% |
| Random Forest | 91.45% | 91.74% | 91.02% | 91.18% |
| Decision Tree | 86.22% | 86.25% | 85.86% | 85.93% |

### Best Classical Model

**Linear SVC** achieved the best overall performance:

- Accuracy: **96.67%**
- Precision: **96.96%**
- Recall: **96.63%**
- F1-score: **96.70%**

Best parameters found through GridSearchCV:

```text
C = 0.5
tol = 0.00005
```

---

## LSTM Models

The second approach used the raw sensor signals instead of the 561 engineered features.

The sensor data was represented as sequences with:

```text
128 time steps
9 sensor channels
```

Several LSTM architectures were tested to compare their performance.

### LSTM Architectures

#### Model 1 — 1-layer LSTM

```text
LSTM(32)
Dropout
Dense(6)
```

Accuracy:

**88.56%**

---

#### Model 2 — 2-layer LSTM (48, 32)

```text
LSTM(48, return_sequences=True)
Batch Normalization
Dropout

LSTM(32)
Dropout

Dense(6)
```

Accuracy:

**92.16%**

This was the best-performing LSTM model.

---

#### Model 3 — 2-layer LSTM (64, 48)

```text
LSTM(64, return_sequences=True)
Batch Normalization
Dropout

LSTM(48)
Dropout

Dense(6)
```

Accuracy:

**88.77%**

---

## LSTM Results

| Model | Accuracy | Precision | Recall | F1-Score | Loss |
|---|---:|---:|---:|---:|---:|
| LSTM 1-layer (32) | 88.56% | 88.96% | 88.46% | 88.53% | 0.420 |
| **LSTM 2-layer (48, 32)** | **92.16%** | **92.56%** | **92.22%** | **92.17%** | **0.232** |
| LSTM 2-layer (64, 48) | 88.77% | 88.94% | 88.97% | 88.84% | 0.323 |

---

## Overall Model Comparison

The final comparison between all models was:

| Model | Accuracy | F1-Score |
|---|---:|---:|
| **Linear SVC** | **96.67%** | **96.70%** |
| Logistic Regression | 96.54% | 96.54% |
| RBF SVM | 96.27% | 96.23% |
| Random Forest | 91.45% | 91.18% |
| **LSTM (48, 32)** | **92.16%** | **92.17%** |
| LSTM (64, 48) | 88.77% | 88.84% |
| LSTM (32) | 88.56% | 88.53% |
| Decision Tree | 86.22% | 85.93% |

---

## Key Results

The main result from the experiments was that the **classical machine learning models performed better than the LSTM models on this dataset**.

The best model was:

> **Linear SVC — 96.67% accuracy and 96.70% F1-score**

The best LSTM was:

> **2-layer LSTM (48, 32) — 92.16% accuracy and 92.17% F1-score**

This means the best classical model performed about **4.5 percentage points better** than the best LSTM in these experiments.

One important observation is that the UCI HAR dataset already provides a large number of engineered features. These features capture useful information from the underlying sensor signals, which likely helped the classical models perform strongly.

The results also show that a more complex neural network does not automatically produce better results. For this particular dataset, the engineered features combined with classical models were more effective than the tested LSTM architectures.

---

## Confusion Matrix Analysis

Confusion matrices were used to examine which activities were being confused with each other.

One of the more noticeable classification challenges was distinguishing between:

- **Sitting**
- **Standing**

These activities have similar sensor characteristics compared with more dynamic activities such as walking.

The classical models generally handled the six activity classes well, while the LSTM models showed more difficulty with the stationary activities.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

### Machine Learning

- Logistic Regression
- Linear SVC
- Support Vector Machine
- Decision Tree
- Random Forest
- GridSearchCV

### Deep Learning

- LSTM
- Dropout
- Batch Normalization
- Dense Layers

### Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- t-SNE

---

## Project Structure

A possible repository structure is:

```text
Human-Activity-Recognition/
│
├── MLPROJECT1.ipynb
├── README.md
├── LICENSE
│
└── dataset/
    └── UCI HAR Dataset/
```

The main analysis and experiments are contained in:

```text
MLPROJECT1.ipynb
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Human-Activity-Recognition
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### 3. Open the notebook

```bash
jupyter notebook MLPROJECT1.ipynb
```

### 4. Run the notebook

Run the cells in order to reproduce the data analysis, model training, evaluation, and comparison.

---

## Limitations

There are a few limitations to this project:

- The experiments use the predefined train/test split provided by the UCI HAR dataset.
- The classical models use the 561 engineered features already provided by the dataset.
- The LSTM models use raw sensor sequences and therefore do not use the engineered 561 features.
- Only a limited number of LSTM architectures were tested.
- The LSTM results can vary slightly between training runs because neural network training involves random initialization and optimization.
- The results are specific to the UCI HAR dataset and may not directly represent performance on other sensor datasets.
- The project focuses on model comparison rather than real-time deployment on a smartphone or edge device.

---

## Conclusion

This project compared classical machine learning models with LSTM-based deep learning for human activity recognition.

The classical models performed better overall, with **Linear SVC achieving the highest accuracy of 96.67%**.

The best LSTM model, a **2-layer LSTM with 48 and 32 units**, achieved **92.16% accuracy**.

The main observation from the experiments is that the engineered features provided by the UCI HAR dataset were highly effective for this classification problem. Because of this, relatively simple classical models were able to outperform the tested LSTM architectures.

The project also shows why model comparison is important. A more complex model is not necessarily a better model for every dataset or problem.

---

## Future Improvements

Some possible next steps for this project would be:

- Test CNN-LSTM architectures
- Experiment with bidirectional LSTMs
- Perform more systematic hyperparameter tuning for the neural networks

---

