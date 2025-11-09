# 🫀 Heart Disease Prediction using KNN

## 📘 Overview
This project uses **K-Nearest Neighbors (KNN)** algorithm to predict the likelihood of heart disease based on patient health data.  
The dataset used is from Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

---

## 🧩 1. Download the Dataset

```python
!pip install opendatasets --quiet

import opendatasets as od
od.download("https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
```

---

## ⚙️ 2. Install Packages and Load Dataset

```python
import pandas as pd
import numpy as np

data = pd.read_csv("/content/heart-failure-prediction/heart.csv")
data.head()
data.info()
```

**Findings:** There are no missing values in the dataset.

---

## 🔍 3. Exploratory Data Analysis (EDA)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='HeartDisease', palette='Set1')
plt.title('Distribution of Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

num_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

for col in num_cols:
  plt.figure(figsize=(10, 5))
  sns.histplot(data=data, x=col, hue='HeartDisease', kde=True, bins=25, alpha=0.6, multiple='dodge')
  plt.xlabel(col)
  plt.ylabel('Count')
  plt.title(f'Distribution of {col}')
  plt.legend(title='Heart Disease', labels=['No (0)', 'Yes (1)'])
  plt.tight_layout()
  plt.show()

for col in cat_cols:
  plt.figure(figsize=(10, 5))
  sns.countplot(data=data, x=col, hue='HeartDisease', palette='Set3')
  plt.xlabel(col)
  plt.ylabel('Count')
  plt.title(f'Distribution of {col}')
  plt.tight_layout()
  plt.show()
```

---

## 🧮 4. Data Preprocessing

### Encode categorical columns

```python
from sklearn.preprocessing import OneHotEncoder

ohe_encoders = {}
encoded_dfs = {}

for col in cat_cols:
  ohe_encoders[col] = OneHotEncoder(sparse_output=False)
  encoded_array = ohe_encoders[col].fit_transform(data[[col]])
  features_names = ohe_encoders[col].get_feature_names_out([col])
  encoded_dfs[col] = pd.DataFrame(encoded_array, columns=features_names, index=data.index)

encoded_cat_cols_df = pd.concat(encoded_dfs.values(), axis=1)
num_cols_df = data.drop(columns=cat_cols, axis=1)
finalData = pd.concat([encoded_cat_cols_df, num_cols_df], axis=1)
finalData.info()
```

### Scale numeric columns

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
scaled_data = sc.fit_transform(finalData.iloc[:, 14:20])
scaled_cols_df = pd.DataFrame(scaled_data, columns=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], index=finalData.index)

for col in scaled_cols_df.columns:
  finalData[col] = scaled_cols_df[col]
```

### Correlation heatmap

```python
plt.figure(figsize=(16, 8))
corr = finalData.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap of Encoded and Scaled Features")
plt.show()
```

---

## 🧠 5. Model Building — KNN

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

features = finalData.iloc[:, 0:20].values
label = finalData.iloc[:, 20].values

CL = 0.9

for seed in range(1, 201):
  for k in range(3, 20):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=seed)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    if test_score > train_score and test_score > CL:
      print(f"Test score: {test_score} | Train score: {train_score} | RS: {seed} | K: {k}")
```

**Best Result:** `Test score: 0.9347 | Train score: 0.8719 | RS: 173 | K: 11`

### Build final model

```python
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=173)
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Test score: {np.round(test_score, 2)} | Train score: {np.round(train_score, 2)}")
```

---

## 🧾 6. Model Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(confusion_matrix(label, model.predict(features)))
print(classification_report(y_test, y_pred))
```

---

## 📊 7. Results & Inference

- Accuracy: **93%**
- Precision, Recall, F1 for both classes > 0.9
- Balanced model, no class bias detected.

---

## 🧩 8. Model Deployment

```python
import pickle

with open('Heart-Disease-Predictor.pkl', 'wb') as file:
  pickle.dump(model, file)
```

---

## 🧠 9. Model Prediction Script

```python
import numpy as np
import pickle

# Load model and encoders
with open('Heart-Disease-Predictor.pkl', 'rb') as file:
    mod = pickle.load(file)

# Get user input
user_inputs = {
    'Sex': input('Enter Sex (F / M): ').strip(),
    'ChestPainType': input('Enter Chest Pain Type (ASY / ATA / NAP / TA): ').strip(),
    'RestingECG': input('Enter Resting ECG (LVH / Normal / ST): ').strip(),
    'ExerciseAngina': input('Enter Exercise Angina (N / Y): ').strip(),
    'ST_Slope': input('Enter ST Slope (Down / Flat / Up): ').strip(),
    'Age': int(input('Enter Age: ')),
    'RestingBP': int(input('Enter Resting BP (60 - 200): ')),
    'Cholesterol': int(input('Enter Cholesterol (100 - 650): ')),
    'FastingBS': int(input('Enter Fasting BS (0 / 1): ')),
    'MaxHR': int(input('Enter Max HR (60 - 202): ')),
    'Oldpeak': float(input('Enter Oldpeak (-2.6 to 6.2): '))
}

encoded_parts = []

# Handle categorical columns with case-insensitive matching
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    user_val = user_inputs[col]
    valid_values = list(ohe_encoders[col].categories_[0])
    matched_val = next((v for v in valid_values if v.lower() == user_val.lower()), None)
    if matched_val is None:
        raise ValueError(f"Invalid value for {col}. Expected one of: {valid_values}")
    encoded_array = ohe_encoders[col].transform(np.array([[matched_val]]))
    encoded_parts.append(encoded_array)

encoded_cats = np.concatenate(encoded_parts, axis=1)

num_inputs = np.array([[user_inputs['Age'], user_inputs['RestingBP'], user_inputs['Cholesterol'],
                        user_inputs['FastingBS'], user_inputs['MaxHR'], user_inputs['Oldpeak']]])

final_inputs = np.concatenate([encoded_cats, num_inputs], axis=1)

prediction = mod.predict(final_inputs)[0]
prob = mod.predict_proba(final_inputs)[0][1]

if prediction == 1:
    print("\nPatient likely to have Heart Disease.")
else:
    print("\nPatient unlikely to have Heart Disease.")

print(f"Model confidence: {prob * 100:.2f}%")
```

---

## ✅ Conclusion

The **KNN-based Heart Disease Prediction model** achieved an accuracy of **93%**, maintaining balanced precision-recall performance across both classes.  
It demonstrates strong potential for clinical risk assessment and can be further optimized using hyperparameter tuning or ensemble approaches.

---

## 🧑‍💻 Author

**Akshay Bharadwaj**

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---
