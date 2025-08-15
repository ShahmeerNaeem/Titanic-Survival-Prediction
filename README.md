# 🚢 Titanic Survival Prediction 
A hands-on machine learning project to predict whether a passenger survived the Titanic disaster, using demographic, travel, and ticket information.  
This project demonstrates **data preprocessing**, **feature engineering**, **model selection**, and **hyperparameter tuning** to compare different algorithms.

---

## 📌 Problem Statement

The Titanic dataset contains passenger details but also a **binary survival label** (`Survived`).  
The goal is to **build and compare models** that predict whether a passenger survived, based on the given attributes.

---

## 🎯 Goal

Predict whether a passenger survived the Titanic disaster, using personal, ticket, and cabin details.

---

## 📊 Dataset Overview

- **Source:** `Titanic-Dataset.csv` (from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic))
- **Target:** `Survived` (1 = survived, 0 = did not survive)
- **Features:**
  - Demographic: `Name`, `Sex`, `Age`
  - Ticket & Cabin: `Ticket`, `Fare`, `Cabin`
  - Travel Details: `Pclass`, `SibSp`, `Parch`, `Embarked`
  - Engineered: `Family_members`

---

## 🛠 Feature Engineering & Data Preprocessing

- **Handling Missing Values:**
  - `Cabin` → extracted first letter, filled missing with `"Unknown"`
  - `Embarked` → filled missing with most frequent value
  - `Age` → filled missing with median

- **Encoding Categorical Variables:**
  - Used `pd.get_dummies()` for `Sex`, `Embarked`, and `Cabin`

- **Engineered Feature:**
```python
titanic_df['Family_members'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
```

## 🔄 Project Workflow

          ┌────────────────────┐
          │   Load Dataset     │
          └─────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │  Data Cleaning     │
          │  (Missing values)  │
          └─────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ Feature Engineering │
          │ (Family_members,    │
          │ Cabin extraction)   │
          └─────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ Encoding Categorical│
          │ Variables (dummies) │
          └─────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ Train-Test Split   │
          └─────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ Model Training &   │
          │ Hyperparameter Tuning│
          └─────────┬──────────┘
                    ↓
          ┌────────────────────┐
          │ Model Evaluation   │
          │ (Accuracy, Precision│
          │ Recall, F1)        │
          └────────────────────┘

## 1️⃣ Import Required Libraries

```import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
```


- pandas → Data loading and manipulation
- sklearn.model_selection → Splitting datasets and cross-validation
- sklearn.linear_model → Logistic Regression model
- sklearn.metrics → Evaluation metrics
- sklearn.ensemble → Random Forest model
- sklearn.tree → Decision Tree model

## 2️⃣ Load Dataset

```titanic_df = pd.read_csv('Titanic-Dataset.csv')```

Reads the dataset into a DataFrame for processing, The file should be in the same directory or provide the full path.

## 3️⃣ Initial Dataset Overview
```
titanic_df.shape
titanic_df.dtypes
```


- .shape → Returns (rows, columns) count.
- .dtypes → Shows data types for each column.

## 4️⃣ Check Missing Values
```
titanic_df.isna().sum()
```


- Identifies columns with missing values.
- Essential before any preprocessing.

##5️⃣ Handle Missing Values & Feature Engineering (Cabin & Embarked)
```
# Extract first letter of 'Cabin' to reduce unique categories
titanic_df['Cabin'] = titanic_df['Cabin'].str[0]


# Fill missing Cabin values with 'Unknown'
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('Unknown')

# Fill missing Embarked values with most frequent value
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0])
```

- Cabin → Only keep first letter to represent cabin group. Missing cabins are marked as 'Unknown'.
- Embarked → Filled using mode() (most common port).

## 6️⃣ Encode Categorical Variables
```
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked', 'Cabin'])
```

- Converts categorical columns into dummy/indicator variables (0/1 encoding).
- This makes them suitable for ML algorithms.

## 7️⃣ Fill Missing Age with Median
```
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
```

- Median is robust to outliers compared to mean.

## 8️⃣ Recheck Missing Values
```
titanic_df.isna().sum()
```

- Confirms there are no more missing values before training.

## 9️⃣ Check Class Distribution
```
print(titanic_df['Survived'].value_counts(normalize=True))
```

- Shows survival rate proportions (class balance).

- Helps decide if special handling for imbalance is needed.

##🔟 Feature Engineering — Family Members
```
titanic_df['Family_members'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
```

- Creates a new feature combining siblings/spouses (SibSp) and parents/children (Parch) plus the passenger themself.

## 1️⃣1️⃣ Define Features (X) & Target (y)
```
X = titanic_df.drop(['PassengerId', 'SibSp', 'Parch', 'Name', 'Survived', 'Ticket'], axis=1)
y = titanic_df['Survived']
```

- X → All predictors except IDs and irrelevant columns.
- y → Target column (Survived).

## 1️⃣2️⃣ Train-Test Split
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=12, stratify=y
)
```

- 30% test set
- Stratify → Maintains class proportions in both sets.

## 1️⃣3️⃣ Define Models
```
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}
```

- Logistic Regression → Simple linear model for classification
- Random Forest → Ensemble of decision trees
- Decision Tree → Tree-based classification

## 1️⃣4️⃣ Hyperparameter Grids
```
params = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['saga', 'liblinear']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
}
```

- Defines parameter ranges for RandomizedSearchCV tuning

## 1️⃣5️⃣ Model Training & Evaluation
```
for name, model in models.items():
    kf = KFold(shuffle=True, random_state=42, n_splits=5)
    rs_cv = RandomizedSearchCV(model, params[name], cv=kf, n_iter=10)
    rs_cv.fit(X_train, y_train)
    
    y_pred = rs_cv.predict(X_test)
    
    acc_sc = accuracy_score(y_test, y_pred)
    pre_sc = precision_score(y_test, y_pred)
    rec_sc = recall_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred)
    
    print(f"{name:<20} | Accuracy: {acc_sc:.2f} | Precision: {pre_sc:.2f} | Recall (Sensitivity): {rec_sc:.2f} | F1 Score: {f1_sc:.2f}")
```

- KFold → 5-fold cross-validation for more reliable evaluation
- RandomizedSearchCV → Random search for hyperparameters
  
Prints performance metrics for each model:

- Accuracy → Overall correctness
- Precision → Correct positive predictions
- Recall → Correctly predicted survivors
- F1 Score → Balance between precision and recall
