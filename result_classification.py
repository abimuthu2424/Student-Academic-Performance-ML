# importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# loading the dataset
data = pd.read_csv("dataset.csv")

# creating pass or fail column
# if final marks are 10 or more -> pass (1)
# else -> fail (0)
data['result'] = data['final_marks'].apply(
    lambda x: 1 if x >= 10 else 0
)

# selecting input columns
X = data[['study_hours', 'attendance', 'internal1', 'internal2']]

# output column (pass or fail)
y = data['result']

# splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# creating logistic regression model
model = LogisticRegression()

# training the model
model.fit(X_train, y_train)

# predicting pass or fail
prediction = model.predict(X_test)

# printing the result
print("Pass (1) or Fail (0) prediction:")
print(prediction)
