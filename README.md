# 🚢 Titanic Survival Prediction

This project applies **Machine Learning** techniques to predict whether a passenger survived the Titanic disaster based on demographic and travel details.  
It uses **data preprocessing**, **feature engineering**, and **hyperparameter tuning** to compare multiple models.

---

## 📌 Project Overview
The goal of this project is to:
- Load and clean the Titanic dataset
- Handle missing values and encode categorical variables
- Engineer new features to improve model performance
- Train multiple machine learning models:
  - Logistic Regression  
  - Random Forest  
  - Decision Tree
- Perform **hyperparameter tuning** using `RandomizedSearchCV` with `KFold` cross-validation
- Evaluate models using **Accuracy, Precision, Recall, and F1-score**

---

## 📊 Dataset
The dataset used is **Titanic-Dataset.csv** containing:
- **PassengerId** – Unique ID for each passenger  
- **Survived** – Target variable (1 = Survived, 0 = Did not survive)  
- **Pclass** – Passenger class (1st, 2nd, 3rd)  
- **Name, Sex, Age** – Demographics  
- **SibSp, Parch** – Number of siblings/spouses and parents/children aboard  
- **Ticket, Fare, Cabin, Embarked** – Ticket details and port of embarkation  

📥 **Dataset Link:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)

---

## ⚙️ Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
