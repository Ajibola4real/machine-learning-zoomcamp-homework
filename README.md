# Employee Attrition Prediction

## Project Overview
This Machine Learning Zoomcamp capstone project predicts
whether an employee is likely to leave a company.

## Dataset
IBM HR Analytics Employee Attrition & Performance dataset.
The dataset contains 1,470 employee records.

## Model
I compared Logistic Regression, Random Forest, and XGBoost.
I selected Balanced Logistic Regression because identifying
employees at risk of leaving was my priority.

## Final Model Performance
- Accuracy: 72.79%
- Precision: 34.58%
- Recall: 78.72%
- F1-score: 48.05%
- ROC-AUC: 84.09%

## Project Files
- train.py: Trains and saves the model
- predict.py: Flask prediction API
- requirements.txt: Python dependencies
- Dockerfile: Container configuration
- employee_attrition_model_from_script.pkl: Trained model

## Run Locally
pip install -r requirements.txt
python predict.py

## Run with Docker
docker build -t employee-attrition-api .
docker run -p 9697:9696 employee-attrition-api

## API
POST /predict

Send employee features as JSON.

The API returns:
- attrition: True or False
- attrition_probability: Probability of leaving

## Conclusion
The model can help identify employees who may be at risk
of leaving. Predictions are estimates and should not be
used as the sole basis for employment decisions.


## Cloud Deployment (Render)

I successfully deployed my Employee Attrition Prediction API
to Render using Docker.

**Live API endpoint:**
https://machine-learning-zoomcamp-homework-keun.onrender.com/predict

The API accepts POST requests containing employee information
in JSON format and returns the predicted attrition probability.

I tested the cloud API successfully and received HTTP status 200.
The prediction matched my local Flask and Docker results.

Note: The free Render service may take some time to respond
after a period of inactivity.
