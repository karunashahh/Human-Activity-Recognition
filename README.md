# Human Activity Recognition (HAR) from Smartphone Sensors

Classifies six physical activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING, LAYING) from smartphone accelerometer and gyroscope data, using the
UCI HAR dataset. Compares classical ML models trained on 561 hand-engineered features
against LSTM models trained directly on raw sensor signal sequences.

## Dataset

UCI HAR Dataset 

https://archive.ics.uci.edu/dataset/364/smartphone+dataset+for+human+activity+recognition+har+in+ambient+assisted+living+aal

After downloading, place it alongside the notebook as:
```
UCI_HAR_Dataset/
├── features.txt
├── activity_labels.txt
├── train/
│   ├── X_train.txt, y_train.txt, subject_train.txt
│   └── Inertial Signals/
└── test/
    ├── X_test.txt, y_test.txt, subject_test.txt
    └── Inertial Signals/
```

## Approach

**Exploratory analysis:** checked for duplicates/nulls, visualized activity distribution
per subject, and examined how acceleration magnitude and gravity-angle features separate
stationary vs. moving activities (t-SNE visualization included).

**Classical ML** (trained on the 561 pre-engineered features in `X_train.txt`):
- Logistic Regression, Linear SVC, RBF-kernel SVM, Decision Tree, Random Forest —
  all tuned via `GridSearchCV`

**Deep learning** (trained on raw 128-timestep, 9-channel sensor sequences):
- Three LSTM architectures compared: 1-layer (32 units), 2-layer (48→32), 2-layer (64→48)
- Trained with `EarlyStopping` on validation loss

## Results

| Model | Accuracy % |
|---|---|
| Linear SVC | 96.67 |
| Logistic Regression | 96.54 |
| RBF SVM | 96.27 |
| Random Forest | 91.45 |
| Decision Tree | 86.22 |
| LSTM (see notebook for per-architecture breakdown) | — |

Full precision/recall/F1 and the LSTM-specific results are in the notebook's model
comparison table and conclusion section.

**Takeaway:** classical ML on carefully engineered features outperformed the LSTM models
trained on raw signals in this comparison — worth discussing in terms of feature
engineering quality vs. end-to-end learning, and the relatively small dataset size for
training a deep sequence model from scratch.

## Project structure

```
MLPROJECT1.ipynb       # full analysis: EDA, classical ML, LSTM, comparison
requirements.txt
LICENSE
.gitignore
README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook MLPROJECT1.ipynb
```
Run all cells in order — later cells depend on variables and imports defined earlier,
so don't skip around.
