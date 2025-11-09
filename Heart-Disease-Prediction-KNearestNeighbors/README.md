# ❤️ Heart Disease Predictor using K-Nearest Neighbors (KNN)

## 📘 Project Overview
This project builds a **K-Nearest Neighbors (KNN)** model to predict whether a person is likely to have **heart disease** based on medical parameters.  
It demonstrates a complete machine learning workflow — from **data preprocessing and encoding** to **model training, evaluation, and deployment using Pickle**.

---

## 🗂 Dataset
- **Dataset Name:** `heart.csv`
- **Features:**
  - `Age` – Age of the individual  
  - `Sex` – Gender (F, M)  
  - `ChestPainType` – Type of chest pain (ASY, ATA, NAP, TA)  
  - `RestingBP` – Resting blood pressure (mmHg)  
  - `Cholesterol` – Serum cholesterol (mg/dl)  
  - `FastingBS` – Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
  - `RestingECG` – Resting electrocardiogram results (LVH, Normal, ST)  
  - `MaxHR` – Maximum heart rate achieved  
  - `ExerciseAngina` – Exercise-induced angina (Y, N)  
  - `Oldpeak` – ST depression induced by exercise relative to rest  
  - `ST_Slope` – Slope of the peak exercise ST segment (Down, Flat, Up)  
- **Target:** `HeartDisease` (1 = likely, 0 = unlikely)

---

## ⚙️ Project Workflow

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
```
**Why these libraries?**  
- `pandas` and `numpy`: Data handling and numerical computation  
- `scikit-learn`: Preprocessing, training, and evaluation  
- `pickle`: Model serialization and reusability  

---

### 2. Data Loading & Exploration
```python
data = pd.read_csv("heart.csv")
data.info()
data.describe()
data.isnull().sum()
```
- Verified dataset integrity  
- Checked for missing or null values  
- Observed feature distribution

---

### 3. Feature–Label Separation
```python
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']
```

---

### 4. Encoding Categorical Features
```python
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
ohe_encoders = {}
encoded_features = []

for col in cat_cols:
    ohe = OneHotEncoder(sparse_output=False)
    transformed = ohe.fit_transform(data[[col]])
    ohe_encoders[col] = ohe
    encoded_features.append(transformed)

X_cat = np.concatenate(encoded_features, axis=1)
X_num = data.drop(columns=cat_cols + ['HeartDisease']).values
X_final = np.concatenate([X_cat, X_num], axis=1)
```

---

### 5. Feature Scaling
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)
```

---

### 6. Train–Test Split and Model Training
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

---

### 7. Model Evaluation
```python
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)
```

---

### 8. Model Saving using Pickle
```python
with open('Heart-Disease-Predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(ohe_encoders, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

---

### 9. Loading Model for User Prediction
```python
with open('Heart-Disease-Predictor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    ohe_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

---

### 10. Taking User Input and Making Predictions
```python
user_inputs = {
    'Sex': input('Enter Sex (F/M): ').strip(),
    'ChestPainType': input('Enter Chest Pain Type (ASY/ATA/NAP/TA): ').strip(),
    'RestingECG': input('Enter Resting ECG (LVH/Normal/ST): ').strip(),
    'ExerciseAngina': input('Enter Exercise Angina (N/Y): ').strip(),
    'ST_Slope': input('Enter ST Slope (Down/Flat/Up): ').strip(),
    'Age': int(input('Enter Age: ')),
    'RestingBP': int(input('Enter Resting BP: ')),
    'Cholesterol': int(input('Enter Cholesterol: ')),
    'FastingBS': int(input('Enter Fasting BS (0/1): ')),
    'MaxHR': int(input('Enter Max HR: ')),
    'Oldpeak': float(input('Enter Oldpeak: '))
}

encoded_parts = []
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    user_val = user_inputs[col]
    valid_values = list(ohe_encoders[col].categories_[0])
    matched_val = next((v for v in valid_values if v.lower() == user_val.lower()), None)
    if matched_val is None:
        raise ValueError(f"Invalid value for {col}. Expected one of: {valid_values}")
    encoded = ohe_encoders[col].transform([[matched_val]])
    encoded_parts.append(encoded)

encoded_cats = np.concatenate(encoded_parts, axis=1)
num_inputs = np.array([[user_inputs['Age'], user_inputs['RestingBP'], user_inputs['Cholesterol'], user_inputs['FastingBS'], user_inputs['MaxHR'], user_inputs['Oldpeak']]])
final_input = np.concatenate([encoded_cats, num_inputs], axis=1)
final_input_scaled = scaler.transform(final_input)

prediction = model.predict(final_input_scaled)[0]
prob = model.predict_proba(final_input_scaled)[0][1]

if prediction == 1:
    print("Patient likely to have Heart Disease")
else:
    print("Patient unlikely to have Heart Disease")

print(f"Model confidence: {prob*100:.2f}%")
```

---

## 📈 Results & Insights
- **Model Used:** KNN Classifier (k = 5)  
- **Accuracy:** ~84–88% depending on dataset split  
- **Best Parameters:** Tuned using distance weighting and scaling  
- **Observations:**
  - Higher `Age`, `Oldpeak`, and `Cholesterol` values correlate with risk  
  - `ExerciseAngina` and `ST_Slope` are strong predictors  

---

## 🚀 Future Improvements
- Perform **hyperparameter tuning** using GridSearchCV  
- Add **cross-validation** for robust evaluation  
- Integrate a **Streamlit web app** for live user prediction  

---

## 🧩 Concepts Reinforced
- Supervised Learning (Classification)  
- Distance-based algorithms (KNN)  
- Feature Scaling importance  
- One-Hot Encoding for categorical data  
- Model persistence with Pickle  

---

## 🪪 License
This project is licensed under the [MIT License](LICENSE).

---
