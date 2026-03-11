# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data
2.Split and Scale the Dataset
3.Train the SVM Model with Hyperparameter Optimization
4.Evaluate the Model Performance

## Program:
```
#Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Load the dataset from the URL

data = pd.read_csv('food_items_binary.csv')
  
#Step 2: Data Exploration
#Display the first few rows and column names for verification

print(data.head())
print(data.columns)

#Step 3: Selecting Features and Target 
#Define relevant features and target column
features= ['Calories', 'Total Fat', 'Saturated Fat','Sugars', 'Dietary Fiber','Protein']
target = 'class' #Assuming 'class' is binary (suitable or not suitable for diabetic patients)

X= data[features]
y= data[target]

#Step 4: Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Step 5: Features Scaling
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Step 6: Model Training with Hyperparameter Tuning using GridSearchCV
#Define the SVM model


svm=SVC()


#Set up hyperparameter grid for tuning
param_grid ={
    'C': [0.1,1,10,100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print("Name: Nikhil Nirmal Kumar")
print("Register Number: 212225230201")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred) 
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="749" height="682" alt="image" src="https://github.com/user-attachments/assets/45859ec3-72fe-4af5-95e9-241a7b41d6d5" />
<img width="548" height="267" alt="image" src="https://github.com/user-attachments/assets/93fc448c-9184-4ddf-918a-ae5b80b67ce6" />
<img width="771" height="594" alt="image" src="https://github.com/user-attachments/assets/cbc773c7-328a-46f1-a30b-73c235987a06" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
