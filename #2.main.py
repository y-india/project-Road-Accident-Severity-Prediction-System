import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset =pd.read_csv(r'D:\#PROJECTS\Road Accident Severity Prediction System\Road.csv\Road_encoded.csv')
df = pd.DataFrame(dataset) 
# print(df.head())





############################################
#5. Spliting Features and Target
############################################


x = df.drop('Accident_severity', axis=1)  # Features
y = df['Accident_severity']               # Target variable

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


############################################
#6. Applying models
############################################



#random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model_1 = RandomForestClassifier(n_estimators=200, random_state=42)
model_1.fit(x_train, y_train)
y_pred = model_1.predict(x_test)

print(classification_report(y_test, y_pred))


print(f"\n\n\n{round(model_1.score(x_test, y_test)*100 , 2)}% accuracy achieved using Random Forest Classifier of test data")
print(f"{round(model_1.score(x_train, y_train)*100 , 2)}% accuracy achieved using Random Forest Classifier of train data")


# importances = model_1.feature_importances_
# feature_names = x.columns
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(10, 10))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()





#XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

model_2 = XGBClassifier(random_state=42)
model_2.fit(x_train, y_train)
y_pred_2 = model_2.predict(x_test)    

# print(classification_report(y_test, y_pred_2))
print(f"\n\n\n{round(model_2.score(x_test, y_test)*100 , 2)}% accuracy achieved using XGBoost Classifier of test data")
print(f"{round(model_2.score(x_train, y_train)*100 , 2)}% accuracy achieved using XGBoost Classifier of train data")




#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model_3 = LogisticRegression(max_iter=1000)
model_3.fit(x_train, y_train)
y_pred_3 = model_3.predict(x_test)

# print(classification_report(y_test, y_pred_3))
print(f"\n\n\n{round(model_3.score(x_test, y_test)*100 , 2)}% accuracy achieved using Logistic Regression of test data")  
print(f"{round(model_3.score(x_train, y_train)*100 , 2)}% accuracy achieved using Logistic Regression of train data")





#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
model_4 = DecisionTreeClassifier(random_state=42)
model_4.fit(x_train, y_train)
y_pred_4 = model_4.predict(x_test)
# print(classification_report(y_test, y_pred_4))
print(f"\n\n\n{round(model_4.score(x_test, y_test)*100 , 2)}% accuracy achieved using Decision Tree Classifier of test data")
print(f"{round(model_4.score(x_train, y_train)*100 , 2)}% accuracy achieved using Decision Tree Classifier of train data")






#SVM Classifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
model_5 = SVC()
model_5.fit(x_train, y_train)
y_pred_5 = model_5.predict(x_test)

# print(classification_report(y_test, y_pred_5))
print(f"\n\n\n{round(model_5.score(x_test, y_test)*100 , 2)}% accuracy achieved using SVM Classifier of test data")
print(f"{round(model_5.score(x_train, y_train)*100 , 2)}% accuracy achieved using SVM Classifier of train data")



#######################################################################################
#for finding best parameter of models -> see -> #3.best_param.py
#######################################################################################