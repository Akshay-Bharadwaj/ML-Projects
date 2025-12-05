# 🎓 University Admission Prediction (Decision Tree & Random Forest)

## 📘 Overview

This project implements **regression** models to predict an applicant's **Chance of Admit** (a probability between 0 and 1) to a graduate program based on various academic and profile metrics.

The primary goal was to compare the performance of a **Decision Tree Regressor** against a robust **Random Forest Regressor** ensemble model for this prediction task.

---

## 🧩 1. Dataset & Features

The project utilizes the **Graduate Admissions** dataset from Kaggle.

### Features:
| Feature | Description | Type |
| :--- | :--- | :--- |
| **GRE Score** | Graduate Record Examination score (out of 340) | Numeric |
| **TOEFL Score** | TOEFL score (out of 120) | Numeric |
| **University Rating** | University rating (out of 5) | Numeric |
| **SOP** | Statement of Purpose strength (out of 5) | Numeric |
| **LOR** | Letter of Recommendation strength (out of 5) | Numeric |
| **CGPA** | Undergraduate GPA (out of 10) | Numeric |
| **Research** | Research experience (0 = No, 1 = Yes) | Numeric |
| **Chance of Admit** | Target variable (Probability, 0 to 1) | Numeric |

---

## 🔍 2. Exploratory Data Analysis (EDA)

Initial analysis confirmed **no missing or duplicate values** in the dataset. Visualizations (Boxplots and KDE plots) showed that most numerical features were approximately **normally distributed** and provided insights into potential correlations with the target variable.

---

## 🧮 3. Data Cleaning and Preprocessing

Thorough preprocessing was applied to prepare the data for the regression models:

1.  **Column Removal:** The redundant **'Serial No.'** column was dropped.
2.  **Outlier Treatment:** The **Interquartile Range (IQR) method** was used on the **'LOR '** column, leading to the removal of **one single outlier** data point to improve model stability.
3.  **Feature Scaling:** All 7 numerical features were scaled using **`StandardScaler`** to standardize them (mean of 0, standard deviation of 1).
4.  **Target Normalization:** The target variable, **'Chance of Admit '**, was normalized between 0 and 1 using **`MinMaxScaler`**.

---

## 🧠 4. Model Building — Decision Tree vs. Random Forest

The dataset was split into training and testing sets (80/20 split) and models were evaluated using 5-fold cross-validation.

### Cross-Validation Comparison:

| Model | Mean Cross-Validation Score (CL) | Observation |
| :--- | :--- | :--- |
| **Decision Tree Regressor** | $\approx 0.54$ | Low score, indicating poor generalization. |
| **Random Forest Regressor** | $\approx 0.78$ | Significantly better, justifying the use of ensemble method. |

### Final Model Training:

The **Random Forest Regressor** was chosen as the final model due to its superior performance and ability to reduce variance (overfitting). Hyperparameters were tuned to avoid the high training score / low testing score scenario, resulting in a **generalized model**:

| Parameter | Value |
| :--- | :--- |
| `n_estimators` | 10 |
| `max_depth` | 3 |
| `random_state` | 120 |

---

## 🧾 5. Model Evaluation

The final Random Forest Regressor achieved strong performance on the unseen test data:

* **Test Score ($\text{R}^2$):** $\mathbf{0.85}$
* **Train Score ($\text{R}^2$):** $0.81$

The scores indicate that the model is **well-generalized** and has a high predictive accuracy, successfully explaining 85% of the variance in the 'Chance of Admit' on the test set.

---

## 🧩 6. Model Deployment

The final trained Random Forest model has been serialized and saved using **Joblib**.

**Model File:** `uni-admission-pred.pkl`

This file can be loaded in a production environment for real-time inference:

```python
import joblib
import numpy as np

# Load the model
final_model = joblib.load('uni-admission-pred.pkl')

# Placeholder for user input (must be scaled before prediction)
# Example inputs (needs to be scaled by the original StandardScaler fit!)
user_input_scaled = np.array([
    # GRE, TOEFL, Univ Rating, SOP, LOR, CGPA, Research (SCALED VALUES)
    [1.81, 1.77, 0.75, 1.12, 1.09, 1.76, 0.87]
])

# Prediction output (will be a scaled value, 0 to 1)
pred_scaled = final_model.predict(user_input_scaled)[0]

# Denormalize the output (using the original MinMaxScaler range)
# ... code for inverse_transform ...

print(f"Predicted Chance of Admission: {np.round(pred_scaled, 2) * 100}%")
```

---

## ✅ Conclusion

The Random Forest-based Admission Prediction model achieved a high $\text{R}^2$ score of $\mathbf{0.85}$ on the test set. The ensemble method effectively leveraged multiple decision trees, leading to a significant improvement over the standalone Decision Tree Regressor and confirming its capability as a strong predictive tool for this regression problem. The successful outlier removal and feature scaling steps contributed to building a robust and generalized model.

---

## 🧑‍💻 Author

**Akshay Bharadwaj**

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---
