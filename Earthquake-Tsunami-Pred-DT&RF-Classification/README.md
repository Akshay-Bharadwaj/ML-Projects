# 🌍 Earthquake & Tsunami Prediction (Decision Tree & Random Forest)

## 📘 Overview

This project implements **classification** models to predict whether an earthquake will trigger a **Tsunami** (binary outcome: 1 for Yes, 0 for No). Using seismic data such as magnitude, intensity, and location, the project compares the performance of a **Decision Tree Classifier** against an ensemble **Random Forest Classifier** to build a reliable risk assessment tool.

---

## 🧩 1. Dataset & Features

The project utilizes the **Global Earthquake & Tsunami Risk Assessment** dataset.

### Features:
| Feature | Description | Type |
| :--- | :--- | :--- |
| **Magnitude** | The earthquake's size on the Richter scale | Numeric |
| **CDI** | Community Decimal Intensity | Numeric |
| **MMI** | Modified Mercalli Intensity | Numeric |
| **Sig** | Significance of the event | Numeric |
| **NST** | Number of seismic stations | Numeric |
| **Dmin** | Minimum distance to epicenter | Numeric |
| **Gap** | Largest azimuthal gap between stations | Numeric |
| **Depth** | Depth of the earthquake (km) | Numeric |
| **Latitude** | Geographic coordinate (N/S) | Numeric |
| **Longitude** | Geographic coordinate (E/W) | Numeric |
| **Tsunami** | **Target variable** (0 = No, 1 = Yes) | Categorical |

---

## 🔍 2. Exploratory Data Analysis (EDA)

Initial analysis confirmed no missing values. **Boxplots** were used to identify outliers across numerical columns. The EDA revealed that the target variable (`tsunami`) was highly **imbalanced**, which was later addressed using SMOTE to ensure the model could accurately identify the minority class.

---

## 🧮 3. Data Cleaning and Preprocessing

The following steps were executed to prepare the data for modeling:

1.  **Feature Selection:** Dropped `Year` and `Month` as they are record-keeping metadata with no predictive impact on seismic mechanics.
2.  **Outlier Removal:** Applied the **Interquartile Range (IQR) method** to filter extreme values in `magnitude`, `mmi`, `sig`, `dmin`, `gap`, and `depth`.
3.  **Handling Imbalance:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset, preventing the model from being biased toward non-tsunami events.
4.  **Feature Scaling:** Standardized the features using **`StandardScaler`** to ensure all metrics contribute equally to the model's decision-making.

---

## 🧠 4. Model Building — Decision Tree vs. Random Forest

Models were evaluated using 5-fold cross-validation to ensure stability.

### Cross-Validation Comparison:

| Model | Mean Cross-Validation Score (CL) | Observation |
| :--- | :--- | :--- |
| **Decision Tree Classifier** | $\approx 0.80$ | Good baseline, but prone to overfitting. |
| **Random Forest Classifier** | $\approx 0.87$ | Significantly more stable and accurate. |

### Final Model Optimization:

To prevent overfitting, hyperparameters were tuned and different random states were tested:

* **Decision Tree:** Achieved **97.6% Test Accuracy** with `max_depth=5` (Random State: 155).
* **Random Forest:** Achieved **97.6% Test Accuracy** with `n_estimators=15` and `max_depth=5` (Random State: 296).



---

## 🧾 5. Model Evaluation

The final models are well-balanced and generalized, showing strong performance on unseen data:

* **Final Accuracy Score:** $\mathbf{89\%}$ (General inference accuracy on unknown data).
* **Generalization:** The high test scores (approx. 97%) relative to training scores indicate that the models effectively learned the patterns of seismic signatures without memorizing the training set.

---

## 🧩 6. Model Deployment

The trained Random Forest model has been serialized using **Joblib** for real-time predictions.

**Model File:** `eq_tsunami_predictor.pkl`

### Real-time Inference Example:

```python
import joblib

# Load the model
final_model = joblib.load('eq_tsunami_predictor.pkl')

# Inputs: mag, cdi, mmi, sig, nst, dmin, gap, depth, lat, lon
inputs = [[7.2, 5.0, 6.0, 850, 120, 1.5, 25, 10, -8.1, 110.2]]
prediction = final_model.predict(inputs)

if prediction[0] == 0:
    print('There is no chance of Tsunami')
else:
    print('Tsunami may occur. Please take safety measures.')

---

## ✅ Conclusion

The Earthquake-Tsunami prediction model demonstrates that classification algorithms, particularly Random Forest, can accurately assess disaster risks. By addressing data imbalance with SMOTE and applying hyperparameter tuning to prevent overfitting, we developed a robust system capable of predicting tsunami occurrences with high precision. Both models achieved a balanced state, ensuring they are reliable for predicting outcomes on new, unknown seismic data.

---

## 🧑‍💻 Author

**Akshay Bharadwaj**

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---
