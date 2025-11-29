!pip install lifelines

#Importaion of libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from lifelines import CoxPHFitter

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, roc_auc_score

#Load of the Dataset

df = pd.read_csv("operational_risk_dataset")

df.head()

#Load of the Dataset

df = pd.read_csv("operational_risk_dataset")

df.head()

#Load of the Dataset

df = pd.read_csv("operational_risk_dataset")

df.head()

#Load of the Dataset

df = pd.read_csv("operational_risk_dataset")

df.head()

#Correlation Heatmap

%matplotlib inline

df.head()

df.shape

df.dtypes

sns.histplot(df["transaction_amount"])

plt.show()	

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))

sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues")

plt.title("Correlation Heatmap")

plt.show()

df.corr()

#Survival Analysis (Cox PH Model)

cox_df = df[[

    "time_to_event", 
    
    "event_occurred",
    
    "transaction_amount",
    
    "failed_attempts",
    
    "system_latency",
    
    "network_errors",
    
    "server_load",
    
]]

cox_model = CoxPHFitter()

cox_model.fit(cox_df, duration_col="time_to_event", event_col="event_occurred")

cox_model.print_summary()

#Plot Survival Curves

cox_model.plot()

plt.title("Cox Model Coefficients")

plt.show()

#Machine Learning Approach 

X = df[[

    "transaction_amount",
    
    "failed_attempts",
    
    "system_latency",
    
    "network_errors",
    
    "server_load",
    
]]

y = df["event_occurred"]

#Train-Test Split	

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42
    
)

#Train Random  forest Classifier

model = RandomForestClassifier(

    n_estimators=300,
    
    random_state=42,
    
    max_depth=8
    
)

model.fit(X_train, y_train)

#Model Evaluation

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:,1]

print("Classification Report:\n", classification_report(y_test, pred))

print("ROC AUC:", roc_auc_score(y_test, prob))

plt.figure(figsize=(8,5))

sns.barplot(x=model.feature_importances_, y=X.columns)

plt.title("Feature Importance for Operational Risk Prediction")

plt.show()

#Predick Risk

sample = pd.DataFrame({

    "transaction_amount": [500],
    
    "failed_attempts": [3],
    
    "system_latency": [80],
    
    "network_errors": [2],
    
    "server_load": [85],
})

print("Predicted risk probability:", model.predict_proba(sample)[0][1])


