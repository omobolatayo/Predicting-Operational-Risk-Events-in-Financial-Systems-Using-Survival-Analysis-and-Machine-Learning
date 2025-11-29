Operational risk events such as system failures, fraud attempts, server overload, and application downtime, pose significant threats to financial institutions. Early prediction of these events helps organizations implement preventive strategies and maintain system reliability.

This project builds a hybrid risk-prediction framework using:
Survival Analysis (Cox Proportional Hazards Model)
Machine Learning (Random Forest Classifier)
Both models complement each other, offering insights into when risk events are likely to occur and how likely they are to occur.
Objectives of the projects are:
Build a synthetic operational risk dataset representing typical financial system behaviors.
Use Cox Proportional Hazards Model to analyze time-to-event and estimate hazard ratios.
Use Random Forest Classifier to predict the probability of an operational risk event.
Compare both approaches and identify key risk drivers.
Provide a reproducible end-to-end notebook for risk modelling.\
The dataset was generated programmatically to simulate realistic operational patterns found in financial institutions, which representing synthetic operational data from a financial system.

**Survival analysis is designed to model time-to-event data.**
It is commonly used in:
Credit default timing
Machine failure prediction
Customer churn timing
Insurance risk modelling
The Cox Proportional Hazards Model estimates the impact of system variables on the hazard rate, which is the risk of an event occurring at time t.
If the hazard ratio for server_load = 1.45, then: A one-unit increase in server load increases the risk of system failure by 45%. This helps identify which system behaviors lead to failures.
 
Machine Learning - Random Forest Classifier
Random Forest is an ensemble machine learning method widely used for fraud detection, credit scoring, and system reliability prediction.
This model provides actionable insights for risk mitigation.

Mostly this hybrid approach mirrors methodologies used in:
Banking risk management, Cloud system reliability, Fraud detection systems, Cybersecurity operations.
Using both models gives a more complete risk-prediction framework (Cox PH - Predicts when a risk event will happen, Random Forest - Predicts if a risk event will happen, Together- Provide robust operational risk intelligence)

**Results Summary**
Survival Analysis Findings
High server load significantly increases the hazard rate.
Network errors accelerate time-to-failure.
System latency is a critical predictor of risk events.
Machine Learning Findings
Random Forest achieved strong predictive performance (e.g., high ROC-AUC).
Top predictors include:
System latency
Server load
Failed attempts
Network errors
