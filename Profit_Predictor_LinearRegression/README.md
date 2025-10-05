# 🧠 Profit Predictor using Linear Regression

## 📘 Project Overview
This project builds a **Linear Regression model** to predict a company’s **profit** based on its **spending patterns** (R&D, Administration, Marketing) and **geographical location (State)**.  
It is an end-to-end **beginner-friendly ML project** covering **data preprocessing, feature encoding, model training, evaluation, and model deployment using Pickle**.

---

## 🗂 Dataset
- **Dataset Name:** `50_Startups.csv`
- **Features:**
  - `R&D Spend` – Investment in research and development  
  - `Administration` – Administrative expenses  
  - `Marketing Spend` – Marketing expenditure  
  - `State` – Categorical variable indicating company location  
- **Target:** `Profit`

---

## ⚙️ Project Workflow

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
```
**Why these libraries?**
- `pandas` & `numpy`: Data manipulation and numerical computations  
- `scikit-learn`: Preprocessing, model creation, and evaluation  
- `pickle`: For saving and reloading the trained model efficiently

---

### 2. Data Loading & Exploration
```python
data = pd.read_csv("50_Startups.csv")
data.info()
```
- Verified no missing or null values.  
- Understood data structure (5 columns).

---

### 3. Feature–Label Separation
```python
features = data.iloc[:, [0,1,2,3]].values
label = data.iloc[:, [4]].values
```
Separated:
- **Features:** Input variables (spending + location)  
- **Label:** Output variable (profit)

---

### 4. Handling Categorical Data (One-Hot Encoding)
`State` is a categorical column, so numerical encoding was needed.

```python
from sklearn.preprocessing import OneHotEncoder
oheState = OneHotEncoder(sparse_output=False)
stateDummy = oheState.fit_transform(data.iloc[:, [3]])
```
- One-Hot Encoding creates binary dummy columns for each category.  
- Prevents giving a false ordinal relationship to categorical data.  
- Concatenated the encoded columns with numerical features to form the final input dataset.

---

### 5. Train–Test Split and Model Training
```python
X_train, X_test, y_train, y_test = train_test_split(finalFeatureSet, label, test_size=0.2, random_state=best_rs)
model = LinearRegression()
model.fit(X_train, y_train)
```
- Dataset split into **80% training** and **20% testing**.  
- **Linear Regression** model fits a line minimizing the sum of squared errors between predicted and actual profit values.

---

### 6. Model Evaluation using Train & Test Scores
```python
train_score = model.score(X_train, y_train)
test_score  = model.score(X_test, y_test)
```
- **Train Score:** Measures how well the model fits the training data.  
- **Test Score:** Measures how well the model generalizes to unseen data.  
- You iterated over multiple random states (1–100) to find a split where the model performs consistently well (test score ≥ CL).

---

### 7. Understanding CL and SL

| Term | Full Form | Explanation | Relevance in This Project |
|------|------------|--------------|----------------------------|
| **SL**(*Significance Level*) -  How much uncertainty you’re willing to accept
| **CL**(*Confidence Level*) - How sure you are that your result is correct

CL = 1 - SL

---

### 8. Model Serialization with Pickle
After training the final model:
```python
pickle.dump(model, open('profit_predictor.pkl', 'wb'))
```

Later, reload it for deployment or predictions:
```python
loaded_model = pickle.load(open('profit_predictor.pkl', 'rb'))
```
Pickle ensures the model can be reused without retraining, making deployment faster.

---

## 📈 Results & Insights
- **Best Random State:** Found experimentally between 1–100.  
- **Test Score ≥ 0.9:** Model achieved strong generalization.  
- **R&D Spend** strongly correlates with profit — the most influential feature.  
- **State** variable had a smaller but observable impact.

---

## 🚀 Future Improvements
- Visualize feature impact using **Seaborn regression plots** or **correlation heatmaps**.  
- Add **user input UI** for real-time profit prediction.

---

## 🧩 Concepts Reinforced
- Linear Regression (Supervised ML Algorithm)  
- Categorical Encoding (One-Hot Encoding)  
- Train–Test Split & Random State Tuning  
- Confidence Level (CL) and Significance Level (SL)  
- Model Evaluation using `.score()`  
- Model Serialization (Pickle)

---
