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
