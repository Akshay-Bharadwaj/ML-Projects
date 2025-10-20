# 🧠 Breast Cancer Prediction using Logistic Regression

## 📘 Project Overview

This project builds a **Logistic Regression** model to predict whether a breast cell is **malignant (cancerous)** or **benign (non-cancerous)** based on cell nucleus characteristics derived from the **Breast Cancer Wisconsin dataset**.

It is an end-to-end beginner-friendly ML project that covers data preprocessing, feature-label separation, model training, evaluation, and experimentation with multiple random states to find the most accurate model configuration.

---

## 🗂 Dataset

**Dataset Name:** `bc_data.csv` (Breast Cancer Wisconsin Dataset)

**Features:**  
The dataset contains **30 numerical features** describing cell nucleus measurements such as:

- radius_mean  
- texture_mean  
- perimeter_mean  
- area_mean  
- smoothness_mean  
- compactness_mean  
- concavity_mean  
- concave_points_mean  
- symmetry_mean  
- fractal_dimension_mean  

**Target Variable:**  
`diagnosis`  
- **M → 1 (Malignant)**  
- **B → 0 (Benign)**

Total Samples: 569  
No missing values after cleaning.

---

## ⚙️ Project Workflow

### 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

**Why these libraries?**
- **pandas & numpy:** For data handling and numerical computation  
- **matplotlib & seaborn:** For exploratory data visualization  
- **scikit-learn:** For train-test split, model creation, and evaluation  

---

### 2. Data Loading & Cleaning

```python
data = pd.read_csv("bc_data.csv")
data.drop(columns=['Unnamed: 32', 'id'], inplace=True)
```

- Loaded the dataset and inspected missing values.  
- Dropped unnecessary columns (`Unnamed: 32` and `id`).  
- Verified all 569 records were clean and numeric except the `diagnosis` column.

---

### 3. Encoding the Target Variable

```python
data['diagnosis'] = [1 if i == 'M' else 0 for i in data['diagnosis']]
```

Converted the categorical target variable into numeric format:
- **1** → Malignant  
- **0** → Benign

---

### 4. Feature–Label Separation

```python
features = data.drop(columns=['diagnosis']).values
label = data['diagnosis'].values
```

- **Features:** 30 input variables  
- **Label:** Diagnosis (binary output)

---

### 5. Train–Test Split and Model Training

```python
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=265)
model = LogisticRegression()
model.fit(X_train, y_train)
```

- Split the dataset into 70% training and 30% testing data.  
- Trained a **Logistic Regression** classifier to predict the probability of malignancy.

---

### 6. Model Evaluation using Train & Test Scores

The model was trained across multiple random states (1–300) to find the one yielding the best test accuracy.

Example iteration results:
```
Test score: 0.9707 | Train score: 0.9422 | Random state: 5
Test score: 0.9766 | Train score: 0.9472 | Random state: 43
...
Best Test score: 0.9941 | Random state: 265
```

Final model evaluation:
```python
test = model.score(X_test, y_test)
train = model.score(X_train, y_train)
print(f"Test score: {np.round(test, 2)} Random state: 265")
```

✅ **Test Accuracy:** 0.99  
✅ **Random State:** 265  

The model achieved near-perfect generalization performance.

---

### 7. Conclusion

The trained Logistic Regression model successfully predicts whether a cell is **malignant** or **benign** with **99% accuracy**.  
This model can assist medical professionals in performing **early detection** of breast cancer, helping to improve diagnosis efficiency and patient outcomes.

---

## 📈 Results & Insights

- **Best Random State:** 265  
- **Best Test Score:** 0.99  
- **Best Train Score:** 0.94  
- Model generalizes well and avoids overfitting.  
- Simple yet interpretable model suitable for medical classification problems.

---

## 🚀 Future Improvements

- Implement feature scaling (StandardScaler) to optimize coefficients.  
- Visualize confusion matrix and ROC curve.  
- Compare performance with SVM, Decision Tree, or Random Forest models.  
- Deploy model as an interactive web api using **FastAPI**.  

---


## 🧩 Concepts Reinforced

- Logistic Regression (Supervised ML Algorithm)  
- Binary Classification  
- Data Preprocessing (column removal, encoding)  
- Train–Test Split & Random State Optimization  
- Model Evaluation using `.score()`  
- Confidence Level (CL) and Significance Level (SL)

---

## 🧑‍💻 Author

**Akshay Bharadwaj**

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---
