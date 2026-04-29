# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Import the required packages and print the present data Print the placement data and salary data. Find the null and duplicate values. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: D.HARITHA
RegisterNumber: 212225040118
*/
```
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

df = pd.read_csv('/content/Placement_Data.csv')
df1 = df.copy()

df1 = df1.drop(['sl_no', 'salary'], axis=1)

le = LabelEncoder()
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 
                    'degree_t', 'workex', 'specialisation', 'status']

for col in categorical_cols:
    df1[col] = le.fit_transform(df1[col])

X = df1.iloc[:, :-1]
y = df1['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Confusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion, display_labels=['Placed', 'Not Placed']
)
disp.plot()

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
<img width="966" height="402" alt="image" src="https://github.com/user-attachments/assets/4f8ab2f4-13cb-4b43-bb80-af211f0102f8" />

<img width="1042" height="647" alt="image" src="https://github.com/user-attachments/assets/37179870-ed2b-4f04-8711-c6e96389e425" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
