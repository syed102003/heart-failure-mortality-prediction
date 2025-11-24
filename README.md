# ❤️ Heart Failure Mortality Prediction using Machine Learning

This repository contains a complete **Machine Learning project** focused on predicting **mortality (DEATH_EVENT)** in heart failure patients using clinical features.  
Two models are implemented — **Logistic Regression** and **Random Forest**, along with full EDA and performance evaluation.

---

## 🚀 Project Overview
Heart failure is one of the leading global causes of mortality.  
This project uses clinical data to predict whether a patient is likely to experience a death event. The goal is to build an interpretable and accurate ML model to assist healthcare analytics.

---

## 🧵 Features
- Full dataset analysis and preprocessing  
- Exploratory Data Analysis (EDA): histograms, boxplots, countplots  
- Feature scaling  
- Model Building: Logistic Regression & Random Forest  
- Evaluation using:
  - Accuracy  
  - Confusion Matrix  
  - Classification Report  
  - ROC AUC Score  
- ROC Curve comparison of both models  

---

## 📂 Dataset
Dataset used: **Heart Failure Clinical Records Dataset**  
Target variable:  
- `DEATH_EVENT` → 0 = Survived, 1 = Death Event  

Features include:
- age  
- anaemia  
- creatinine_phosphokinase  
- diabetes  
- ejection_fraction  
- high_blood_pressure  
- platelets  
- serum_creatinine  
- serum_sodium  
- sex  
- smoking  
- time  

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Logistic Regression  
- Random Forest  

---

## 🧪 Code Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Step 2: Load Dataset
df = pd.read_csv("Heart_failure_clinical_records_dataset.csv")
df.head()

# Step 3: Data Overview
print("Shape:", df.shape)
print(df.info())
print(df.describe())
print(df['DEATH_EVENT'].value_counts())

# Step 4: Check for Null Values
print(df.isnull().sum())

# Step 5: Exploratory Data Analysis (EDA)
sns.countplot(x='DEATH_EVENT', data=df)
plt.title('Death Event Distribution')
plt.show()

sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data=df)
plt.title("CPK vs Death Event")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 6: Feature and Target Split
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Step 7: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Step 10: Evaluation - Logistic Regression
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("ROC AUC Score:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Step 11: Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Step 12: Evaluation - Random Forest
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Step 13: ROC Curve Plot
fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

## 📈 Results
- Both models give strong predictive performance  
- Random Forest typically achieves higher accuracy and AUC  
- ROC Curve clearly shows model comparison  

---

## 🔮 Future Improvements
- Hyperparameter tuning for Random Forest  
- Try Gradient Boosting / XGBoost  
- Build a Streamlit dashboard  
- Deploy model using Flask/FastAPI  

---

## 🧑‍💻 Author
**Syed Abdul Manan**  
B.Tech AI & ML  

---

## ⭐ Support
If you like this project, give it a **star** ⭐ on GitHub!


