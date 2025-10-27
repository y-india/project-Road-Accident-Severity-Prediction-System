import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(r'D:\#PROJECTS\Road Accident Severity Prediction System\Road.csv\Road_encoded.csv')
df = pd.DataFrame(dataset)


X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Load RandomForest
rf_model_loaded = joblib.load(r'D:\#PROJECTS\Road Accident Severity Prediction System\random_forest_model_of_accident_project.pkl')

# Load XGBoost
xgb_model_loaded = joblib.load(r'D:\#PROJECTS\Road Accident Severity Prediction System\XGBoost_model_of_accident_project.pkl')

# Load SVC
svc_model_loaded = joblib.load(r'D:\#PROJECTS\Road Accident Severity Prediction System\SVM_model_of_accident_project.pkl')



# Predictions
y_pred_rf = rf_model_loaded.predict(X_test)
y_pred_xgb = xgb_model_loaded.predict(X_test)
y_pred_svc = svc_model_loaded.predict(X_test)

X_new = [
    [1,5,2,0.1,4,0.1,0.1,1,0.1,0.1,0.1,0.1,3,0.1,0.1,2,0.1,0.1,0.1],
    [2,3,4,0.1,2,0.1,0.5,1,0.1,0.2,0.1,0.1,3,0.1,0.2,1,0.2,0.1,0.5],
    [1,3,3,0.1,4,0.1,0.1,5,0.1,0.2,0.1,0.1,3,0.1,0.2,2,0.2,0.1,0.7]
]



predictions = {
    'RandomForest': rf_model_loaded.predict(X_new),
    'XGBoost': xgb_model_loaded.predict(X_new),
    'SVC': svc_model_loaded.predict(X_new)
}





df_pred = pd.DataFrame(predictions)
print("Predicted Accident Severity for New Data:\n")
print(df_pred)
