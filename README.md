# 🚗 Road Accident Severity Prediction System  

### 🔍 Predicting Accident Severity Using Machine Learning  

---

## 📘 Project Overview

This project — **Road Accident Severity Prediction System** — aims to predict the **severity of road accidents** based on real-world data from India.  
The dataset contains multiple factors such as driver’s age, experience, vehicle type, road conditions, light conditions, and more.  
The goal is to assist in understanding how various features contribute to accident severity and help authorities focus on prevention.

---

## 🧩 Problem Statement

Road accidents are one of India’s major causes of injuries and fatalities.  
The problem is to **predict the level of accident severity** (Slight, Serious, or Fatal) given multiple conditions like weather, road surface, driver profile, and vehicle information.

---

## 💡 Solution Approach

To solve this, a **Machine Learning classification approach** is used.  
The workflow consists of several key steps:

1. **Data Collection & Understanding**  
   - Dataset: `road.csv` (real-world Indian accident dataset)
   - Columns include driver info, vehicle info, road/environmental conditions, and accident details.

2. **Data Cleaning & Exploration**  
   - Performed in **`#1data_work.ipynb`**  
   - Initially explored nulls, unique values, and distributions.
   - In `why_not_droping.py`, dropping rows led to ~75% data loss → decided *not to drop* rows.
   - Then filled nulls using **SimpleImputer** from `sklearn.impute`.

3. **Feature Engineering & Encoding**  
   - Dropped irrelevant columns (explained in `why_i_droped.txt`).
   - Encoded categorical features using `.map()`.
   - Saved cleaned dataset as **`road_encoded.csv`**.

4. **Model Building & Training**  
   - Done in **`#2.main.py`**
   - Split data into **features (X)** and **target (y)**.
   - Models applied:
     - ✅ Random Forest Classifier  
     - ✅ XGBoost Classifier  
     - ✅ Logistic Regression  
     - 🔄 Decision Tree Classifier  
     - 🔄 SVM Classifier

5. **Hyperparameter Tuning**  
   - Performed in **`#3.best_param.py`**
   - Tuned models using **RandomizedSearchCV** and **GridSearchCV**.
   - Best-performing models identified:  
     - Support Vector Machine (SVM)  
     - Random Forest Classifier (RFC)  
     - XGBoost Classifier (XGB)
   - Saved trained models using **Joblib**.

6. **Model Evaluation & Comparison**  
   - Visualized model accuracies and confusion matrices using `matplotlib` & `seaborn`.

7. **Final Model Selection**  
   - Chose **Random Forest Classifier** for deployment based on balance of accuracy and interpretability.  
   - Model file: `random_forest_model_of_accident_project.pkl`

8. **Prediction Testing**  
   - Implemented in **`#4PREDICTION.py`**  
   - Tested predictions on new encoded data points.

9. **Streamlit App Deployment**  
   - Created in **`#5app.py`**
   - Features:
     - Model loading and UI creation
     - Dropdowns for all encoded features (no manual number input)
     - Collapsible section for full encoding reference
     - Prediction output displayed with color-coded result:  
       - 🟢 Slight Injury  
       - 🟡 Serious Injury  
       - 🔴 Fatal Injury

---

## 🎨 Streamlit App Features

- Interactive web UI built using **Streamlit**
- All categorical values shown as dropdowns for easy selection
- Clean UI with sections:
  - **Title, Caption & Description**
  - **Encoding Reference Expander**
  - **Input Form (20+ dropdowns)**
  - **Prediction Display**

---

## 🚦 Example Test Cases

You can try the following example cases in the app:

| Case | Description | Expected Result |
|------|--------------|----------------|
| 1 | City Minor Crash — Taxi, 18–30 yrs, Dry road, Daylight | 🟢 Slight Injury |
| 2 | Highway Wet Road Collision — Lorry, 31–50 yrs, Raining | 🟡 Serious Injury |
| 3 | Foggy Rural Night Crash — Motorcycle, Over 51 yrs, Drunk | 🔴 Fatal Injury |

---

## 📂 Repository Structure

```
📁 Road Accident Severity Prediction System/
│
├── road.csv # Original dataset
├── road_encoded.csv # Cleaned and encoded dataset
│
├── #1data_work.ipynb # Initial exploration and cleaning
├── why_not_droping.py # Null handling experiment
├── why_i_droped.txt # Columns removed explanation
├── #2.main.py # Model training and comparison
├── #3.best_param.py # Hyperparameter tuning
├── #4PREDICTION.py # Model testing and prediction
├── #5app.py # Final Streamlit application
├── random_forest_model_of_accident_project.pkl # Saved ML model
│-----and many more files...
└── README.md # Project documentation
```



---

## 🧠 Skills & Technologies Used

**Languages & Libraries**
- Python 🐍  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- Joblib  
- Streamlit  

**Concepts Applied**
- Data Cleaning & Encoding  
- Feature Engineering  
- Model Training & Evaluation  
- Hyperparameter Tuning  
- Data Visualization  
- Streamlit Web App Development  

---

## 📊 Model Used

### 🎯 Random Forest Classifier
```python
model_1 = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=None,
    bootstrap=False,
    random_state=42
)
model_1.fit(X_train, y_train)
```

---
### 👨‍💻 Author
## Yuvraj

| 🇮🇳 India

Aspiring Data Scientist & Machine Learning Enthusiast

Skilled in Python, Streamlit, and Machine Learning

Passionate about solving real-world problems with data

📫 Let’s connect & collaborate!




### 📧 **Contact Me:**  

✉️ y.india.main@gmail.com

