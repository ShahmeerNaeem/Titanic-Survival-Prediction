# 🚢 Titanic Survival Prediction — End-to-End ML Project

## 📌 Overview
This project predicts whether a passenger survived the Titanic disaster using machine learning techniques. It follows a complete ML workflow: data cleaning, feature engineering, model selection, hyperparameter tuning, and evaluation.

## 📂 Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Features include passenger demographics, ticket information, cabin details, and more.

## 🛠 Workflow
1. **Data Preprocessing**
   - Imputed missing values in `Age`, `Cabin`, and `Embarked`
   - Extracted first letter from `Cabin` (cabin section)
   - One-hot encoded categorical variables
   - Filled missing values in numeric features with median

2. **Feature Engineering**
   - Created `Family_members` feature (`SibSp` + `Parch` + 1)
   - Dropped irrelevant columns (`PassengerId`, `Name`, `Ticket`)

3. **Modeling**
   - Compared:
     - Logistic Regression
     - Random Forest Classifier
     - Decision Tree Classifier
   - Used `RandomizedSearchCV` with `KFold` cross-validation for tuning

4. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

## 📊 Results (example output)
| Model              | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.82     | 0.77      | 0.72   | 0.74     |
| Random Forest      | 0.85     | 0.81      | 0.78   | 0.79     |
| Decision Tree      | 0.80     | 0.75      | 0.74   | 0.74     |

## 📦 Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn

## 🚀 Next Steps
- Wrap preprocessing + modeling into a Scikit-learn Pipeline
- Add model persistence (`joblib`)
- Deploy as a simple API or Streamlit app

## 📜 License
This project is open-source under the MIT License.

## 🔗 Repository
[GitHub Repo](https://github.com/yourusername/titanic-ml-project)
