from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from .logistic_regression import LogisticRegression
from .linear_regression import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

from django.http import HttpResponse
from io import BytesIO
import base64
import urllib


def home(request):
    return render(request, 'home.html')
def index(request):
    return render(request, 'index.html')
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

    model1 = LinearRegression()
    model1.fit(X_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    pred1 = model1.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    # Map the predicted probability onto a scale from 0 to 100
    print(pred)
    print(pred1)



    #diabetes_level = np.around(pred * 100, 2)
    #print(diabetes_level)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    logReg = LogisticRegression(lr=0.1, num_iter=10000)
    logReg.fit(X_train_std, y_train)
    y_pred = logReg.predict(X_test_std)

    logReg1 = LinearRegression(lr=0.1, num_iter=10000)
    logReg1.fit(X_train_std,y_train)
    y_pred1 = logReg1.predict(X_test_std)

    logistic_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    linear_pred = [1 if p >= 0.5 else 0 for p in y_pred1]

    logistic_accuracy = accuracy_score(y_test,logistic_pred )
    print(f"Logistic Accuracy: {logistic_accuracy}")

    linear_accuracy = accuracy_score(y_test,linear_pred )
    print(f"Linear Accuracy: {linear_accuracy}")

    if logistic_accuracy > linear_accuracy:
        accuracy1 = logistic_accuracy
    elif logistic_accuracy < linear_accuracy:
        accuracy1 = linear_accuracy
    else:
        accuracy1 = logistic_accuracy
    print("Accuracy:", accuracy1 * 100)
     #Compute binary predictions using a threshold
    # logistic_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    # linear_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred1]
    #
    # # Compute confusion matrices
    # linear_cm = confusion_matrix(y_test, logistic_pred_binary)
    # logistic_cm = confusion_matrix(y_test, linear_pred_binary)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    #
    # sns.heatmap(logistic_cm, annot=True, cmap='Blues', fmt='d', ax=axes[0])
    # axes[0].set_title('Logistic Regression Confusion Matrix')
    # axes[0].set_xlabel('Predicted')
    # axes[0].set_ylabel('True')
    # sns.heatmap(linear_cm, annot=True, cmap='Blues', fmt='d', ax=axes[1])
    # axes[1].set_title('Linear Regression Confusion Matrix')
    # axes[1].set_xlabel('Predicted')
    # axes[1].set_ylabel('True')
    # buf = BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # string = base64.b64encode(buf.read())
    # uri2 = urllib.parse.quote(string)

    # Vary threshold and calculate accuracy for logistic regression
    thresholds = np.arange(0, 1.05, 0.05)
    logistic_accuracies = []
    for threshold in thresholds:
        binary_pred = [1 if p >= threshold else 0 for p in y_pred]
        accuracy = accuracy_score(y_test, binary_pred)
        logistic_accuracies.append(accuracy)
    # Vary threshold and calculate accuracy for linear regression
    linear_accuracies = []
    for threshold in thresholds:
        binary_pred = [1 if p >= threshold else 0 for p in y_pred1]
        accuracy = accuracy_score(y_test, binary_pred)
        linear_accuracies.append(accuracy)

    # Generate the accuracy comparison plot
    plt.plot(thresholds, logistic_accuracies, label='Logistic Regression')
    plt.plot(thresholds, linear_accuracies, label='Linear Regression')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri3 = urllib.parse.quote(string)



# Interpret the level of diabetes severity based on the scale
    if np.any(pred >= 0.8):
        result1 = "Severe Diabetes"
        Message = "Visit Hospital and Consult Doctor ASAP."


    elif np.any(pred >= 0.5):
        result1 = "Moderate Diabetes"
        Message = "Visit Hospital and Consult"

    elif np.any(pred >= 0.2):

        result1 = "Mild Diabetes"
        Message = "Manage diabetes with proper care."
    else:
        result1 = "Low or No Risk of Diabetes"
        Message = "Stay Healthy"

    return render(request, 'predict.html', {"result2":result1, "diabetes_level": pred, 'result3':Message,
  'accuracy':accuracy1*100, 'uri3':uri3, })

