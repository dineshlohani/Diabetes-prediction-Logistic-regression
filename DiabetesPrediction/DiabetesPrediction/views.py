from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from .logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')

def name(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv(r'C:\Users\dinul\diabetes.csv')
    X = data.drop("Outcome", axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    # Map the predicted probability onto a scale from 0 to 100
    print(pred)
    #diabetes_level = np.around(pred * 100, 2)
    #print(diabetes_level)

    # Interpret the level of diabetes severity based on the scale
    if pred >= 0.8:
        result1 = "Severe Diabetes"
        Message = "Visit Hospital and Consult Doctor ASAP."

    elif pred >= 0.50 :
        result1 = "Moderate Diabetes"
        Message = "Visit Hospital and Consult"
    elif pred >= 0.20 :
        result1 = "Mild Diabetes"
        Message = "Manage diabetes with proper care."
    else:
        result1 = "Low or No Risk of Diabetes"
        Message = "Stay Healthy"

    return render(request, 'predict.html', {"result2":result1, "diabetes_level": pred, 'result3':Message})

